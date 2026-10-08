# Third-party Notices

The root [MIT license](LICENSE) covers original code and documentation in this repository. It does not replace the rights or license notices attached to external code, paper figures, model weights, or datasets.

## InQuant paper figures

`docs/assets/figure-1.png` and `docs/assets/figure-3.png` reproduce the diagrams from Figures 1 and 3 of *InQuant: In-Place Mixed-Precision KV Cache Quantization via Saliency-Aware Neighbor-Slot Reuse*.

The figures were extracted from pages 2 and 3 of the supplied manuscript, `23436_InQuant_In_Place_Mixed_P.pdf`, for the paper overview in the README. They retain the original rights of their respective authors and are not relicensed under this repository's MIT license. The full manuscript is not bundled. The supplied copy does not provide a public author list or a verifiable DOI; this repository does not assign one or claim to be the authors' official release.

## ZipCache

- Upstream: https://github.com/ThisisBillhe/ZipCache
- Pinned commit: `8833f675a938b019fccc531bfdf932abe6e622ad`
- Location: `third_party/ZipCache/`
- Project license: [MIT](third_party/ZipCache/LICENSE), Copyright (c) 2024 ThisisBillhe.
- The Qwen adapter uses the unmodified `zipcache/models/CompressUtils/compress_function.py`. Its SHA256 is `a35c5b0aa6acaea4350e6ef9af13d817c27ff7a22be81658812154762c1595a7`.

The upstream `modeling_llama.py` and `modeling_mistral.py` files retain Hugging Face, EleutherAI, and Mistral AI copyright notices and Apache-2.0 headers. Those notices are preserved, and a copy of the [Apache-2.0 license](third_party/licenses/Apache-2.0.txt) is included.

## Installed dependencies and downloaded assets

PyTorch, Triton, Transformers, vLLM, KVpress, math-verify, and other installed dependencies retain their upstream licenses. The vLLM extension uses plugin registration and subclassing; it does not bundle a modified vLLM installation.

Qwen2.5-7B-Instruct weights are downloaded from `Qwen/Qwen2.5-7B-Instruct` on Hugging Face. GSM8K and MATH500 are fetched from `openai/gsm8k` and `HuggingFaceH4/MATH-500`. They are not included in the source archive. Their use and redistribution are governed by their respective upstream terms. Pinned download commands are provided in the main README.

## RULER and AIME benchmark assets

RULER v1 generators are obtained separately from [NVIDIA/RULER](https://github.com/NVIDIA/RULER) at commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` (Apache-2.0). The source archive does not vendor these generators or their downloaded corpora. The local scorer implements the published substring matching protocol, including fractional target recall. Paul Graham essays, SQuAD, HotpotQA, and NLTK tokenizer resources retain their own rights and terms; the preparation manifest records their source URLs and downloaded hashes.

[AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026) is downloaded from MathArena at revision `d2de22f3c656b4f56cf8981212186377d1e23bc3`. The dataset is published under **CC BY-NC-SA 4.0**, not the repository's MIT license. Raw questions, answers, and generated benchmark datasets are not included in the source ZIP.
