"""Deterministic RNA/structure features shared by training and inference.

No labels, IDs, species, or split-specific information enter the feature matrix.
Fit make_preprocessor() on training features only.
"""
from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

SEQUENCE_ALPHABET = "ACGU"
STRUCTURE_ALPHABET = ".()"
NUMERIC_FEATURES = ["mfe", "mfe_per_nt", "sequence_length", "gc_fraction", "noncanonical_fraction"]


def _kmers(alphabet, k):
    return ["".join(chars) for chars in product(alphabet, repeat=k)]


FEATURE_NAMES = [
    f"{prefix}_{k}mer_{word}"
    for prefix, alphabet in [("rna", SEQUENCE_ALPHABET), ("structure", STRUCTURE_ALPHABET)]
    for k in (1, 2, 3)
    for word in _kmers(alphabet, k)
] + NUMERIC_FEATURES

FEATURE_CONFIG = {
    "schema_version": 1,
    "sequence_normalization": "uppercase_then_T_to_U; no trimming",
    "sequence_alphabet": SEQUENCE_ALPHABET,
    "structure_alphabet": STRUCTURE_ALPHABET,
    "k_values": [1, 2, 3],
    "kmer_order": "product(alphabet, repeat=k); RNA then structure; increasing k",
    "kmer_normalization": "overlapping counts divided by valid windows for each k; zero if none",
    "structure_orientation": "left_to_right; do not reverse or symmetrize",
    "numeric_features": NUMERIC_FEATURES,
    "gc_denominator": "number of A,C,G,U nucleotides",
    "noncanonical_denominator": "full normalized sequence length",
    "nonfinite_mfe": "NaN; median imputation fitted on training only",
    "preprocessing": ["SimpleImputer(strategy=median, keep_empty_features=True)"],
    "feature_count": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "label_mapping": {"0": 0, "1": 1},
    "excluded_inputs": ["id", "species", "row_order", "split_membership"],
}


def save_feature_config(path):
    Path(path).write_text(json.dumps(FEATURE_CONFIG, indent=2) + "\n")


def validate_feature_config(path):
    """Reject artifact/extractor mismatches rather than silently changing features."""
    config = json.loads(Path(path).read_text())
    if config != FEATURE_CONFIG:
        raise ValueError("Feature configuration does not match this version of features.py")
    return config


def normalize_sequence(value):
    if not isinstance(value, str) or not value:
        raise ValueError("sequence_rna must be a nonmissing, nonempty string")
    return value.upper().replace("T", "U")


def load_samples(input_dir):
    """Read only input/samples.tsv. This loader never reads labels.csv."""
    path = Path(input_dir) / "samples.tsv"
    # Keep RNA text such as 'NA' intact rather than interpreting it as a missing token.
    frame = pd.read_csv(path, sep="\t", dtype={"id": str}, keep_default_na=False)
    required = {"id", "sequence_rna", "structure", "mfe"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required samples.tsv columns: {sorted(missing)}")
    if frame["id"].isna().any() or frame["id"].eq("").any():
        raise ValueError("IDs must be nonmissing, nonempty strings")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate IDs in samples.tsv")
    return frame


def load_labeled_split(split_dir):
    """Optional training helper: split-local 1:1 join preserving input row order."""
    split_dir = Path(split_dir)
    samples = load_samples(split_dir / "input")
    labels = pd.read_csv(split_dir / "labels.csv", dtype={"id": str})
    if "label" not in labels and "numeric_label" in labels:
        labels = labels.rename(columns={"numeric_label": "label"})
    if not {"id", "label"}.issubset(labels.columns):
        raise ValueError("labels.csv must contain id and label (or numeric_label)")
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError("Missing or duplicate label IDs")
    if set(samples["id"]) != set(labels["id"]):
        raise ValueError("Input and label IDs do not match exactly within this split")
    joined = samples.merge(labels[["id", "label"]], on="id", how="left", sort=False,
                           validate="one_to_one")
    if not joined["label"].isin([0, 1]).all():
        raise ValueError("Labels must be numeric 0 or 1")
    joined["label"] = joined["label"].astype(np.int64)
    return joined


_KMER_MAPS = {
    alphabet: [(k, {word: i for i, word in enumerate(_kmers(alphabet, k))})
               for k in (1, 2, 3)]
    for alphabet in (SEQUENCE_ALPHABET, STRUCTURE_ALPHABET)
}


def _frequencies(text, alphabet):
    result = []
    for k, index in _KMER_MAPS[alphabet]:
        counts = np.zeros(len(index), dtype=np.float64)
        valid = 0
        for start in range(len(text) - k + 1):
            position = index.get(text[start:start + k])
            if position is not None:
                counts[position] += 1
                valid += 1
        if valid:
            counts /= valid
        result.extend(counts)
    return result


def _validate_structure(structure, length):
    if not isinstance(structure, str) or not structure:
        raise ValueError("structure must be a nonmissing, nonempty string")
    if len(structure) != length:
        raise ValueError("sequence_rna and structure lengths disagree")
    if set(structure) - set(STRUCTURE_ALPHABET):
        raise ValueError("structure contains characters outside '.', '(', ')'")
    depth = 0
    for char in structure:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            raise ValueError("Unbalanced dot-bracket structure")
    if depth:
        raise ValueError("Unbalanced dot-bracket structure")


def _extract_original_features(samples):
    """Return float64 (n_samples, 128), allowing NaN only in MFE features.

    Operates on any nonempty RNA window length, with explicit length features.
    Empty, absent or malformed sequences/structures fail instead of being imputed.
    """
    required = {"sequence_rna", "structure", "mfe"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    matrix = np.empty((len(samples), 128), dtype=np.float64)
    mfe_values = pd.to_numeric(samples["mfe"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_values[~np.isfinite(mfe_values)] = np.nan
    for i, (raw_seq, structure) in enumerate(zip(samples["sequence_rna"], samples["structure"])):
        try:
            seq = normalize_sequence(raw_seq)
            n = len(seq)
            _validate_structure(structure, n)
            canonical = sum(seq.count(base) for base in SEQUENCE_ALPHABET)
            gc = (seq.count("G") + seq.count("C")) / canonical if canonical else 0.0
            row = _frequencies(seq, SEQUENCE_ALPHABET)
            row += _frequencies(structure, STRUCTURE_ALPHABET)
            row += [mfe_values[i], mfe_values[i] / n, n, gc, (n - canonical) / n]
            matrix[i] = row
        except ValueError as exc:
            raise ValueError(f"Invalid sample at input row {i}: {exc}") from exc
    return matrix


def make_preprocessor():
    """Returns an UNFITTED preprocessor; fit exclusively on training data."""
    return SimpleImputer(strategy="median", keep_empty_features=True)


# Expanded representation, iteration 2. Original 128 columns above are unchanged.
ORIGINAL_FEATURE_NAMES = FEATURE_NAMES.copy()


def parse_structure(structure, length):
    """Return partners, post-character depths, sorted pairs and nesting children."""
    _validate_structure(structure, length)
    partner = np.full(length, -1, dtype=int)
    depths = np.zeros(length, dtype=int)
    stack, pairs, children, roots = [], [], {}, []
    for i, char in enumerate(structure):
        if char == '(':
            children[i] = []
            if stack:
                children[stack[-1]].append(i)
            else:
                roots.append(i)
            stack.append(i)
        elif char == ')':
            left = stack.pop()
            partner[left] = i
            partner[i] = left
            pairs.append((left, i))
        depths[i] = len(stack)
    return partner, depths, sorted(pairs), children, roots


def _runs(mask):
    lengths, current = [], 0
    for x in mask:
        if x:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _expanded(seq, structure):
    n = len(seq)
    partner, depth, pairs, children, roots = parse_structure(structure, n)
    paired = partner >= 0
    out = {}
    def put(name, value):
        out[name] = float(value)
    def summary(prefix, values, stats, scale=1):
        a = np.asarray(values, dtype=float)
        for stat in stats:
            if not len(a):
                value = 0
            elif stat.startswith('q'):
                value = np.quantile(a, float(stat[1:]))
            else:
                value = getattr(np, stat)(a)
            put(prefix + '_' + stat, value / scale)
    def length_bins(prefix, a, bounds):
        for name, lo, hi in bounds:
            put(prefix + '_' + name, sum(lo <= x <= hi for x in a) / n)

    windows = [(f'bin25_{b}', b * n // 8, (b+1)*n//8) for b in range(8)]
    windows += [(f'center{size}', max(0,(n-size)//2), min(n,(n+size)//2))
                for size in (40,80,120,160)]
    for name, lo, hi in windows:
        s = seq[lo:hi]
        canonical = sum(s.count(b) for b in SEQUENCE_ALPHABET)
        for b in SEQUENCE_ALPHABET:
            put(name+'_'+b, s.count(b)/canonical if canonical else 0)
        put(name+'_gc', (s.count('G')+s.count('C'))/canonical if canonical else 0)
        for key, mask in [('paired',paired), ('open',np.array(list(structure))=='('),
                          ('close',np.array(list(structure))==')')]:
            put(name+'_'+key, np.mean(mask[lo:hi]) if hi>lo else 0)
        summary(name+'_depth',depth[lo:hi],['mean','max','std'],n)
        p = partner[lo:hi]
        put(name+'_external_pair', np.mean((p>=0)&((p<lo)|(p>=hi))) if hi>lo else 0)
        distances = np.abs(p-np.arange(lo,hi))[p>=0]
        summary(name+'_partner_distance',distances,['mean'],n)

    for b in range(4):
        lo, hi = b*n//4, (b+1)*n//4
        for prefix, text, alphabet, ks in [('rna',seq[lo:hi],SEQUENCE_ALPHABET,(1,2)),
                                         ('structure',structure[lo:hi],STRUCTURE_ALPHABET,(1,2,3))]:
            vals = _frequencies(text,alphabet)
            names = [f'bin50_{b}_{prefix}_{k}mer_{word}' for k in (1,2,3)
                     for word in _kmers(alphabet,k)]
            limit = sum(len(alphabet)**k for k in ks)
            for name, value in zip(names[:limit], vals[:limit]):
                put(name,value)

    put('topology_pair_count',len(pairs)/n)
    summary('topology_depth',depth,['max','mean','std','q0.25','q0.5','q0.75','q0.9'],n)
    summary('topology_span',[j-i for i,j in pairs],['mean','std','min','max','q0.25','q0.5','q0.75','q0.9'],n)
    dots = _runs(~paired)
    put('topology_dot_run_count',len(dots)/n)
    summary('topology_dot_run',dots,['mean','std','max'],n)
    length_bins('topology_dot_run',dots,[('1',1,1),('2',2,2),('3_4',3,4),('5_8',5,8),('ge9',9,np.inf)])
    for name, mask in [('paired',paired),('unpaired',~paired)]:
        a = _runs(mask)
        put('topology_'+name+'_run_count',len(a)/n)
        summary('topology_'+name+'_run',a,['mean','max'],n)

    pairset = set(pairs)
    stems = []
    for i,j in pairs:
        if (i-1,j+1) not in pairset:
            length = 1
            while (i+length,j-length) in pairset:
                length += 1
            stems.append((length,i,j))
    put('topology_stem_count',len(stems)/n)
    lengths = [x[0] for x in stems]
    summary('topology_stem_length',lengths,['mean','std','max'],n)
    for k in (3,5,10):
        put(f'topology_stem_ge{k}',sum(x>=k for x in lengths)/n)
    longest = sorted(stems,key=lambda x:(-x[0],x[1]))[0] if stems else (0,0,0)
    length,i,j = longest
    put('topology_longest_pair_fraction',length/len(pairs) if pairs else 0)
    for key,val in [('left',i),('right',j),('midpoint',(i+j)/2),('span',j-i),
                    ('center_distance',abs((i+j)/2-(n-1)/2) if stems else 0)]:
        put('topology_longest_'+key,val/n)

    put('topology_root_count',len(roots)/n)
    childcounts = [len(x) for x in children.values()]
    for name, condition in [('zero',lambda x:x==0),('one',lambda x:x==1),('multi',lambda x:x>=2)]:
        put('topology_children_'+name,sum(condition(x) for x in childcounts)/n)
    put('topology_children_max',max(childcounts,default=0)/n)
    loops, sums, diffs = [], [], []
    bulges = internals = 0
    for i,j in pairs:
        ch = children[i]
        if not ch:
            loops.append(j-i-1)
        elif len(ch)==1:
            k = ch[0]
            l = partner[k]
            a,b = k-i-1,j-l-1
            sums.append(a+b)
            diffs.append(abs(a-b))
            bulges += (a>0)!=(b>0)
            internals += (a>0) and (b>0)
    put('topology_terminal_count',len(loops)/n)
    summary('topology_terminal_length',loops,['mean','std','min','max'],n)
    length_bins('topology_terminal',loops,[('0_3',0,3),('4_8',4,8),('9_15',9,15),('ge16',16,np.inf)])
    summary('topology_gap_sum',sums,['mean','std','max'],n)
    summary('topology_gap_difference',diffs,['mean','std','max'],n)
    put('topology_bulges',bulges/n)
    put('topology_internal_loops',internals/n)

    canonical_pairs = [(seq[i]+seq[j],j-i) for i,j in pairs
                       if seq[i] in SEQUENCE_ALPHABET and seq[j] in SEQUENCE_ALPHABET]
    denom = len(canonical_pairs)
    for word in _kmers(SEQUENCE_ALPHABET,2):
        put('pair_composition_'+word,sum(w==word for w,d in canonical_pairs)/denom if denom else 0)
    categories = [('AU_UA',{'AU','UA'}),('CG_GC',{'CG','GC'}),('GU_UG',{'GU','UG'}),
                  ('other',set(_kmers(SEQUENCE_ALPHABET,2))-{'AU','UA','CG','GC','GU','UG'})]
    for name,words in categories:
        spans = [d for w,d in canonical_pairs if w in words]
        put('pair_composition_'+name+'_fraction',len(spans)/denom if denom else 0)
        summary('pair_composition_'+name+'_distance',spans,['mean'],n)
    put('pair_composition_straddle',sum(i<n/2<=j for i,j in pairs)/len(pairs) if pairs else 0)

    counts = np.zeros((4,8))
    eligible = 0
    for i in range(1,n-1):
        if seq[i] in SEQUENCE_ALPHABET:
            mask = 4*int(paired[i-1])+2*int(paired[i])+int(paired[i+1])
            counts[SEQUENCE_ALPHABET.index(seq[i]),mask] += 1
            eligible += 1
    for b,base in enumerate(SEQUENCE_ALPHABET):
        for mask in range(8):
            put(f'joint_triplet_{base}_{mask:03b}',counts[b,mask]/eligible if eligible else 0)
    return out


_EXPANDED_TEMPLATE = _expanded('A'*200,'.'*200)
FEATURE_NAMES = ORIGINAL_FEATURE_NAMES + list(_EXPANDED_TEMPLATE)
FEATURE_CONFIG = dict(FEATURE_CONFIG, schema_version=2, feature_count=len(FEATURE_NAMES),
    feature_names=FEATURE_NAMES, original_feature_count=128,
    source='iteration_1/data_representation/features.py (original 128 columns unchanged)',
    added_feature_rules={
        'positions':'zero-based; depth immediately after processing each character',
        'windows':'8 equal 25-nt bins, 4 equal 50-nt bins for length 200; centered windows 40/80/120/160 clipped to input length',
        'normalization':'all added lengths, depths and topology counts divided by full sequence length; fractions use their stated eligible populations',
        'stems':'maximal consecutive nested pair chains; longest tie broken by smallest outer left coordinate',
        'pair_categories':'canonical paired endpoints only, excluding N/noncanonical endpoints from all composition denominators',
        'triplet':'mask bits left,center,right; 1=paired; denominator eligible canonical central positions excluding endpoints',
        'undefined':'zero; population standard deviation (ddof=0); linear quantiles',
        'external_pair':'fraction of ALL window positions paired to an endpoint outside the window',
        'no_pairs':'all pair/stem/loop summaries and coordinates zero',
        'center':'longest stem center distance uses (n-1)/2; straddling uses i<n/2<=j',
    }, feature_groups={
        'original':ORIGINAL_FEATURE_NAMES,
        'positional':[k for k in _EXPANDED_TEMPLATE if k.startswith(('bin','center'))],
        'topology':[k for k in _EXPANDED_TEMPLATE if k.startswith('topology')],
        'pair_composition':[k for k in _EXPANDED_TEMPLATE if k.startswith('pair_composition')],
        'joint_triplet':[k for k in _EXPANDED_TEMPLATE if k.startswith('joint_triplet')],
    })


def _extract_iteration2_features(samples):
    """Deterministic float64 matrix; only original MFE columns may contain NaN.

    No fitted state or labels are read. Supports bounded-memory caller batching.
    """
    original = _extract_original_features(samples)
    matrix = np.empty((len(samples),643),dtype=np.float64)
    matrix[:,:128] = original
    for row,(seq,structure) in enumerate(zip(samples['sequence_rna'],samples['structure'])):
        values = _expanded(normalize_sequence(seq),structure)
        if list(values) != list(_EXPANDED_TEMPLATE):
            raise ValueError('Expanded feature order mismatch')
        matrix[row,128:] = list(values.values())
    if not np.isfinite(matrix[:,128:]).all():
        raise ValueError('Nonfinite expanded features')
    return matrix


# Schema v3: retain all iteration-2 columns verbatim, then append fine structure.
_ITERATION2_EXTRACT = _extract_iteration2_features
ITERATION2_FEATURE_NAMES = list(FEATURE_NAMES)
POSITION_FEATURE_NAMES = [f'fine_position_{i:03d}_{stat}' for i in range(200)
                          for stat in ('depth', 'signed_partner_displacement', 'orientation')]
BIN_FEATURE_NAMES = [f'fine_bin10_{b:02d}_{stat}' for b in range(20)
                    for stat in ('paired_fraction', 'opening_fraction', 'closing_fraction',
                                 'mean_depth', 'max_depth', 'mean_absolute_partner_displacement')]
FEATURE_NAMES = ITERATION2_FEATURE_NAMES + POSITION_FEATURE_NAMES + BIN_FEATURE_NAMES
FEATURE_CONFIG.update(
    schema_version=3, feature_count=len(FEATURE_NAMES), feature_names=FEATURE_NAMES,
    previous_schema_feature_count=643,
    fine_structure_rules={
        'required_length': 200,
        'parser': 'balanced dot-bracket; partner=-1 if unpaired; depth after processing each character',
        'position_order': 'i=0..199, then depth, signed_partner_displacement, orientation',
        'depth': 'd[i]/200',
        'signed_partner_displacement': '(p[i]-i)/200 if p[i]>=0 else 0',
        'orientation': "1 for (, -1 for ), 0 for .",
        'bin_order': 'b=0..19; positions [10*b,10*b+10); six statistics in feature-name order',
        'paired_fraction': 'number of positions with p[i]>=0 divided by 10',
        'opening_fraction': 'number of ( divided by 10',
        'closing_fraction': 'number of ) divided by 10',
        'mean_depth': 'mean(d[i])/200 over all 10 positions',
        'max_depth': 'max(d[i])/200 over all 10 positions',
        'mean_absolute_partner_displacement': 'mean(abs(p[i]-i))/200 over paired bin positions; zero if none; partners from full window',
    })
FEATURE_CONFIG['feature_groups'] = dict(FEATURE_CONFIG['feature_groups'],
    fine_position=POSITION_FEATURE_NAMES, fine_bin10=BIN_FEATURE_NAMES)


def _fine_structure(structure):
    if len(structure) != 200:
        raise ValueError('Fine structural schema requires exactly 200 positions')
    partner, depth, _, _, _ = parse_structure(structure, 200)
    paired = partner >= 0
    displacement = np.where(paired, partner - np.arange(200), 0) / 200.0
    orientation = np.array([1 if c == '(' else -1 if c == ')' else 0 for c in structure])
    positions = np.column_stack((depth / 200.0, displacement, orientation)).ravel()
    bins = []
    for lo in range(0, 200, 10):
        sl = slice(lo, lo+10)
        mask = paired[sl]
        bins.extend([mask.mean(), (orientation[sl] == 1).mean(),
                     (orientation[sl] == -1).mean(), depth[sl].mean()/200.0,
                     depth[sl].max()/200.0,
                     np.abs(displacement[sl][mask]).mean() if mask.any() else 0.0])
    return np.concatenate((positions, bins))


def extract_features(samples):
    for seq, structure in zip(samples['sequence_rna'], samples['structure']):
        if len(normalize_sequence(seq)) != 200 or not isinstance(structure, str) or len(structure) != 200:
            raise ValueError('Schema v3 requires sequence and structure lengths exactly 200; no padding/truncation')
    # The legacy function uses the global FEATURE_NAMES to allocate its matrix.
    # Compute its unchanged 643-column implementation with an explicit allocation size.
    original = _extract_iteration2_features(samples)
    matrix = np.empty((len(samples), 1363), dtype=np.float64)
    matrix[:, :643] = original
    for row, structure in enumerate(samples['structure']):
        matrix[row, 643:] = _fine_structure(structure)
    if not np.isfinite(matrix[:, 643:]).all():
        raise ValueError('Nonfinite fine structural features')
    return matrix


# Schema v4: append only deterministic structural geometry.
import hashlib
_ITERATION3_EXTRACT = extract_features
INHERITED_FEATURE_NAMES = list(FEATURE_NAMES)
GEOMETRY_POSITION_NAMES = [f'geometry_position_{i:03d}_{stat}' for i in range(200)
    for stat in ('pair_midpoint', 'stack_length')]
LOOP_STATS = ['present', 'midpoint', 'interior_length', 'opening', 'closing',
              'opening_depth', 'stack_length', 'outermost_opening', 'outermost_closing', 'ancestor_count']
TERMINAL_LOOP_NAMES = [f'terminal_loop_{rank}_{stat}' for rank in range(5) for stat in LOOP_STATS]
CENTER_RADII = [5,10,20,30,40,60]
CENTER_LOOP_NAMES = [f'center_loop_r{r}_{stat}' for r in CENTER_RADII
    for stat in ('midpoint_count', 'interior_fraction', 'spanning_pairs', 'contained_loops')]
FEATURE_NAMES = INHERITED_FEATURE_NAMES + GEOMETRY_POSITION_NAMES + TERMINAL_LOOP_NAMES + CENTER_LOOP_NAMES
FEATURE_CONFIG = dict(FEATURE_CONFIG, schema_version=4, feature_count=len(FEATURE_NAMES),
    feature_names=FEATURE_NAMES, inherited_feature_count=1363,
    structural_geometry={
        'coordinates': 'zero-based; L=200; post-character depth; orientation unchanged',
        'position_order': 'i=0..199: (i+partner[i])/(2L), maximal contiguous stack pair count/L; both zero if unpaired',
        'stack': 'maximal chain (a,b),(a+1,b-1),...; same total length for every member pair',
        'terminal_loop': 'pair with no child pairs; interior consists only of dots (may be empty)',
        'loop_order': 'sort by (abs((a+b)/2-99.5),a,b); retain five; zero pad ten values per missing record',
        'record_order': LOOP_STATS,
        'record_values': ['1','(a+b)/(2L)','(b-a-1)/L','a/L','b/L','post_depth[a]/L','stack_pairs/L',
            'outermost_ancestor_open/L (self if no ancestor)', 'outermost_ancestor_close/L (self if no ancestor)', 'enclosing_ancestor_count/L'],
        'radii': CENTER_RADII,
        'interval': '[100-r,100+r)',
        'summary_order': ['terminal loop midpoint count/L', 'fraction of interval positions in terminal-loop interiors',
            'count of pairs with a<100-r and b>=100+r divided by L', 'count of terminal-loop pairs with both endpoints inside interval/L'],
        'undefined': 'zero'},
    feature_groups=dict(FEATURE_CONFIG['feature_groups'], geometry_position=GEOMETRY_POSITION_NAMES,
                        terminal_loop_records=TERMINAL_LOOP_NAMES, centered_loop_summaries=CENTER_LOOP_NAMES))
FEATURE_CONFIG['feature_names_sha256'] = hashlib.sha256(json.dumps(FEATURE_NAMES, separators=(',', ':')).encode()).hexdigest()


def structural_geometry(structure):
    partner, depth, pairs, children, roots = parse_structure(structure, 200)
    stack_lengths = {}
    for a,b in pairs:
        if a > 0 and partner[a-1] == b+1:
            continue
        members=[]
        x,y=a,b
        while x < y and partner[x] == y:
            members.append((x,y))
            x,y=x+1,y-1
        for pair in members:
            stack_lengths[pair]=len(members)
    parents = {child: opening for opening,kids in children.items() for child in kids}
    loops = sorted([(a,b) for a,b in pairs if not children[a]],
                   key=lambda ab:(abs((ab[0]+ab[1])/2-99.5),ab[0],ab[1]))
    positions=np.zeros((200,2), dtype=np.float64)
    for a,b in pairs:
        positions[a]=positions[b]=((a+b)/400.,stack_lengths[(a,b)]/200.)
    records=np.zeros((5,10), dtype=np.float64)
    for rank,(a,b) in enumerate(loops[:5]):
        outer=a
        ancestors=0
        while outer in parents:
            ancestors+=1
            outer=parents[outer]
        records[rank]=[1,(a+b)/400.,(b-a-1)/200.,a/200.,b/200.,depth[a]/200.,
                       stack_lengths[(a,b)]/200.,outer/200.,partner[outer]/200.,ancestors/200.]
    interiors=np.zeros(200,dtype=bool)
    for a,b in loops:
        interiors[a+1:b]=True
    summaries=[]
    for r in CENTER_RADII:
        lo,hi=100-r,100+r
        summaries.extend([sum(lo <= (a+b)/2 < hi for a,b in loops)/200.,
                          interiors[lo:hi].mean(),
                          sum(a<lo and b>=hi for a,b in pairs)/200.,
                          sum(lo<=a<hi and lo<=b<hi for a,b in loops)/200.])
    return np.concatenate((positions.ravel(),records.ravel(),summaries))


def extract_features(samples):
    inherited = _ITERATION3_EXTRACT(samples)
    matrix = np.empty((len(samples),1837), dtype=np.float64)
    matrix[:,:1363]=inherited
    for row,structure in enumerate(samples['structure']):
        matrix[row,1363:]=structural_geometry(structure)
    if not np.isfinite(matrix[:,1363:]).all():
        raise ValueError('Nonfinite structural geometry features')
    return matrix
