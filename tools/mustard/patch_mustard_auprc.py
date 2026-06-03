#!/usr/bin/env python
from pathlib import Path


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match for MuStARD patch block, found {count}: {old[:80]!r}")
    return text.replace(old, new)


path = Path("mustard_src/src/utilities/python/train_CNN.py")
text = path.read_text()

text = replace_once(
    text,
    "from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger\n",
    "from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback\n",
)
text = replace_once(
    text,
    "from Files import Format\n",
    "from Files import Format\n\n"
    "def average_precision(labels, scores):\n"
    "\tlabels = np.asarray(labels).astype(int)\n"
    "\tscores = np.asarray(scores).astype(float)\n"
    "\tif labels.size == 0 or np.sum(labels) == 0:\n"
    "\t\treturn 0.0\n"
    "\torder = np.argsort(-scores, kind=\"mergesort\")\n"
    "\tsorted_labels = labels[order]\n"
    "\tprecision = np.cumsum(sorted_labels) / (np.arange(sorted_labels.size) + 1)\n"
    "\treturn float(np.sum(precision * sorted_labels) / np.sum(sorted_labels))\n\n"
    "class ValidationAUPRCCallback(Callback):\n"
    "\tdef __init__(self, validation_x, validation_y, positive_class_index):\n"
    "\t\tsuper(ValidationAUPRCCallback, self).__init__()\n"
    "\t\tself.validation_x = validation_x\n"
    "\t\tself.validation_y = validation_y\n"
    "\t\tself.positive_class_index = positive_class_index\n\n"
    "\tdef on_epoch_end(self, epoch, logs=None):\n"
    "\t\tlogs = logs if logs is not None else {}\n"
    "\t\tpredictions = self.model.predict(self.validation_x, verbose=0)\n"
    "\t\tscores = predictions[:, self.positive_class_index] if predictions.ndim > 1 else predictions.ravel()\n"
    "\t\tlabels = np.argmax(self.validation_y, axis=1) == self.positive_class_index\n"
    "\t\tlogs[\"val_auprc\"] = average_precision(labels, scores)\n"
    "\t\tprint(\" - val_auprc: %.4f\" % logs[\"val_auprc\"])\n",
)
text = replace_once(
    text,
    "train_x = []\n\ttrain_y = the_one[\"train\"][\"labels\"]\n\ttest_x = []\n\ttest_y = the_one[\"test\"][\"labels\"]\n\tvalid_x = []\n\tvalid_y = the_one[\"valid\"][\"labels\"]\n",
    "train_x = []\n\ttrain_y = the_one[\"train\"][\"labels\"]\n\ttest_x = []\n\ttest_y = the_one[\"test\"][\"labels\"]\n\tvalid_x = []\n\tvalid_y = the_one[\"valid\"][\"labels\"]\n\tpositive_class_index = int(np.argmax(valid_y[0]))\n",
)
text = replace_once(
    text,
    "mcp = ModelCheckpoint(filepath = tmp_output_directory + \"/CNNonRaw.hdf5\",\n"
    "\t\t\t\tverbose = 0,\n"
    "\t\t\t\tsave_best_only = True)\n"
    "\tearlystopper = EarlyStopping(monitor = 'val_loss', \n"
    "\t\t\t\t\tpatience = 40,\n"
    "\t\t\t\t\tmin_delta = 0,\n"
    "\t\t\t\t\tverbose = 1,\n"
    "\t\t\t\t\tmode = 'auto')\n",
    "mcp = ModelCheckpoint(filepath = tmp_output_directory + \"/CNNonRaw.hdf5\",\n"
    "\t\t\t\tverbose = 0,\n"
    "\t\t\t\tmonitor = 'val_auprc',\n"
    "\t\t\t\tmode = 'max',\n"
    "\t\t\t\tsave_best_only = True)\n"
    "\tearlystopper = EarlyStopping(monitor = 'val_auprc', \n"
    "\t\t\t\t\tpatience = 20,\n"
    "\t\t\t\t\tmin_delta = 0,\n"
    "\t\t\t\t\tverbose = 1,\n"
    "\t\t\t\t\tmode = 'max')\n",
)
text = replace_once(
    text,
    "\tcsv_logger = CSVLogger(tmp_output_directory + \"/CNNonRaw.log.csv\", \n"
    "\t\t\t\tappend=True, \n"
    "\t\t\t\tseparator='\\t')\n",
    "\tcsv_logger = CSVLogger(tmp_output_directory + \"/CNNonRaw.log.csv\", \n"
    "\t\t\t\tappend=True, \n"
    "\t\t\t\tseparator='\\t')\n"
    "\tauprc_callback = ValidationAUPRCCallback(valid_x[0] if len(modes) == 1 else valid_x, valid_y, positive_class_index)\n",
)
callback_count = text.count("callbacks = [mcp, earlystopper, csv_logger]")
if callback_count != 2:
    raise RuntimeError(f"Expected two MuStARD fit callback lists, found {callback_count}")
text = text.replace(
    "callbacks = [mcp, earlystopper, csv_logger]",
    "callbacks = [auprc_callback, mcp, earlystopper, csv_logger]",
)

path.write_text(text)
