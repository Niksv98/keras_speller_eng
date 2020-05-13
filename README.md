# Sequence-to-Sequence Learning for Spelling Correction

This repository contains a Keras implementation of an encoder-decoder LSTM architecture for sequence-to-sequence spelling correction. The character-level spell checker is trained on unigram tokens derived from a vocabulary of more than ~~466k~~ 33k unique English words. After ~~12~~ one hour of training, the speller achieves an accuracy performance of ~~96.7%~~ 97.6% on a validation set comprised of more than 26k tokens.

```
Input sentence:
> The rabbit holV ewnt straight on liek a tnnel ofr some way any then dipped suddnely down so suddnenly tat Alice had nobt a moment to think aPout stopipng herself before she found hersefl falling dow a verZy deeup wLell

Decoded sentence:
> The rabbit hole went straight on like a tunnel for some way any then dipped suddenly drown so suddenly tat Alice had nob a moment to think Pout stopping herself before she found herself falling down a very deep well

Target sentence:
> The rabbit hole went straight on like a tunnel for some way and then dipped suddenly down so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well
```
