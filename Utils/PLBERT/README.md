---
license: mit
datasets:
- styletts2-community/multilingual-phonemes-10k-alpha
language:
- fr
- en
- es
- ca
- de
- el
- fa
- fi
- pt
- pl
- ru
- sv
- uk
- zh
---

# Multilingual PL-BERT checkpoint

The checkpoint open-sourced here is trained by Papercup using the open-source PL-BERT model found here https://github.com/yl4579/PL-BERT. It is trained to be supported by StyleTTS2, which can be found here: https://github.com/yl4579/StyleTTS2. You can see in the model card the languages that it has been trained on (the languages correspond to the crowdsourced dataset found here https://huggingface.co/datasets/styletts2-community/multilingual-phonemes-10k-alpha).

Notable differences compared to the default PL-BERT checkpoint and config available [here](https://github.com/yl4579/StyleTTS2/tree/main/Utils/PLBERT):
* Because we are working with many languages, we are using a different tokenizer now: `bert-base-multilingual-cased`.
* The PL-BERT model was trained on the data obtained from `styletts2-community/multilingual-phonemes-10k-alpha` for 1.1M iterations.
* The `token_maps.pkl` file has changed (also open-sourced here).
* We have changed the `util.py` file to deal with an error when loading `new_state_dict["embeddings.position_ids"]`.

## How do I train StyleTTS2 with this new PL-BERT checkpoint?

* Simply create a new folder under `Utils` in your StyleTTS2 repository. Call it, for example, `PLBERT_all_languages`. 
* Copy paste into it `config.yml`, `step_1100000.t7` and `util.py`.
* Then, in your StyleTTS2 config file, change `PLBERT_dir` to `Utils/PLBERT_all_languages`. You will also need to change your import as such:
  * Change `from Utils.PLBERT.util import load_plbert`
  * To `from Utils.PLBERT_all_languages.util import load_plbert`
  * Alternatively, you can just replace the relevant files in `Utils/PLBERT` and not have to change any code.
* Now, you need to create train and validation files. You will need to use `espeak` to create a file in the same format as the ones that exist in the `Data` folder of the StyleTTS2 repository. Careful! You will need to change the `language` argument to phonemise your text if it's not in English. You can find the correct language codes [here](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md). For example, Latin American Spanish is `es-419`

Voila, you can now train a multilingual StyleTTS2 model!

Thank you to Aaron (Yinghao) Li for these contributions.