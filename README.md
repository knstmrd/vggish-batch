# VGGish batch audio processing

Batch processing for the [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) model, which produces 128-dimensional embeddings for 960ms long audio fragments.

The code here is intended to be dropped into the same directory, with the new `vggish_input.py` replacing the original one, and a new file (`vggish_batch_inference.py`) doing the actual batch processing.

## Input correction

The following two input correction routines have been added to `vggish_input.py`:

1. Since the VGGish model requires a sufficiently long audio input to operate, if a file's length is less than 960ms, it is repeated until its length exceed 960ms (so a 800ms long audio file will be repeated twice, a 300ms long fragment will be repeated four times, etc.)
1. If an audio file's length is 0, a 960-ms long array of silence is used.

## Batch processing

To process multiple files, one should call `$ python vggish_batch_inference.py --wav_files input.txt`, where `input.txt` is a text file containing a path to a wav file on each line.
The output (a numpy array) is written to a `.npy` file (by default it is written to `output.npy`). For each audiofile, the mean of the embeddings along the time axis is taken (so if a file produces 3 embeddings of length 128, the mean of these embeddings is written to the file, as 128-dimensional vector).

## License

Apache 2.0