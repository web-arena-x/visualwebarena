# X-WebArena: The Unified Repository for WebArena and VisualWebArena

## News
* [08/05/2024]: Added an [Amazon Machine Image](environment_docker/README.md#pre-installed-amazon-machine-image) that pre-installed all VWA (and WA) websites so that you don't have to!
* [03/08/2024]: Added the [agent trajectories](https://drive.google.com/file/d/1-tKz5ByWa1-jwtejiFgxli8fZcBPZgAE/view?usp=sharing) of our GPT-4V + SoM agent on the full set of 910 VWA tasks.
* [02/14/2024]: Added a [demo script](run_demo.py) for running the GPT-4V + SoM agent on any task on an arbitrary website.
* [01/25/2024]: GitHub repo released with tasks and scripts for setting up the VWA environments.
* [12/21/2023] We release the recording of trajectories performed by human annotators on ~170 tasks. Check out the [resource page](./resources/README.md#12212023-human-trajectories) for more details.
* [11/3/2023] Multiple features!
  * Uploaded newest [execution trajectories](./resources/README.md#1132023-execution-traces-from-our-experiments-v2)
  * Added [Amazon Machine Image](./environment_docker/README.md#pre-installed-amazon-machine-image) that pre-installed all websites so that you don't have to!
  * [Zeno](https://zenoml.com/) x WebArena which allows you to analyze your agents on WebArena without pain. Check out this [notebook](./scripts/webarena-zeno.ipynb) to upload your own data to Zeno, and [this](https://hub.zenoml.com/project/9db3e1cf-6e28-4cfc-aeec-1670cac01872/WebArena%20Tester/explore?params=eyJtb2RlbCI6ImdwdDM1LWRpcmVjdCIsIm1ldHJpYyI6eyJpZCI6NzQ5MiwibmFtZSI6InN1Y2Nlc3MiLCJ0eXBlIjoibWVhbiIsImNvbHVtbnMiOlsic3VjY2VzcyJdfSwiY29tcGFyaXNvbk1vZGVsIjoiZ3B0NC1jb3QiLCJjb21wYXJpc29uQ29sdW1uIjp7ImlkIjoiYTVlMDFiZDUtZTg0NS00M2I4LTllNDgtYTU4NzRiNDJjNjNhIiwibmFtZSI6ImNvbnRleHQiLCJjb2x1bW5UeXBlIjoiT1VUUFVUIiwiZGF0YVR5cGUiOiJOT01JTkFMIiwibW9kZWwiOiJncHQzNS1kaXJlY3QifSwiY29tcGFyZVNvcnQiOltudWxsLHRydWVdLCJtZXRyaWNSYW5nZSI6WzAsMV0sInNlbGVjdGlvbnMiOnsibWV0YWRhdGEiOnt9LCJzbGljZXMiOltdLCJ0YWdzIjpbXX19) page for browsing our existing results!
* [10/24/2023] We re-examined the whole dataset and fixed the spotted annotation bugs. The current version ([v0.2.0](https://github.com/web-arena-x/webarena/releases/tag/v0.2.0)) is relatively stable and we don't expect major updates on the annotation in the future. The new results with better prompts and the comparison with human performance can be found in our [paper](https://arxiv.org/abs/2307.13854)
* [8/4/2023] Added the instructions and the docker resources to host your own WebArena Environment. Check out [this page](environment_docker/README.md) for details.
* [7/29/2023] Added [a well commented script](minimal_example.py) to walk through the environment setup.

## Install
```bash
# Python 3.10 (or 3.11, but not 3.12 cause 3.12 deprecated distutils needed here)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```

You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```
pytest -x
```

## Setup Environment
> [!IMPORTANT]
> The demo sites are only for browsing purpose to help you better understand how the websites look like. To ensure the correct evaluation, please setup your own websites.
Since WebArena and VisualWebArena uses a different set of websites, the concrete commands are slightly different. 
- [WebArena environment setup](README_WA.md#webarena-environment-setup)
- [VisualWebArena environment setup](README_VWA.md#visualwebarena-environment-setup)

## End-to-End Evaluation
1. Set up API keys.

If using OpenAI models, set a valid OpenAI API key (starting with `sk-`) as the environment variable:
```
export OPENAI_API_KEY=your_key
```

If using Gemini, first install the [gcloud CLI](https://cloud.google.com/sdk/docs/install). Configure the API key by authenticating with Google Cloud:
```
gcloud auth login
gcloud config set project <your_project_name>
```
2. Launch the evaluation. 
For example, to run WebArena CoT
```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \ # this is the reasoning agent prompt we used in the paper
  --test_start_idx 25 \
  --test_end_idx 26 \
  --model gpt-3.5-turbo \
  --test_config_base_dir=config_files/wa/test_webarena \
  --result_dir <your_result_dir>
```
This script will run the 25th example in WebArena with GPT-3.5 agent. The trajectory will be saved in <your_result_dir>/25.html

To reproduce the other baselines from VisualWebArena, please check these a few commands [1](README_VWA.md#gpt-35-captioning-baseline)[2](README_VWA.md#gpt-4v--som-agent).


### Demo
![Demo](media/find_restaurant.gif)

We have also prepared a demo for you to run the agents on your own task on an arbitrary webpage. An example is shown above where the agent is tasked to find the best Thai restaurant in Pittsburgh.

After following the setup instructions above and setting the OpenAI API key (the other environment variables for website URLs aren't really used, so you should be able to set them to some dummy variable), you can run the GPT-4V + SoM agent with the following command:
```bash
python run_demo.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --start_url "https://www.amazon.com" \
  --image "https://media.npr.org/assets/img/2023/01/14/this-is-fine_wide-0077dc0607062e15b476fb7f3bd99c5f340af356-s1400-c100.jpg" \
  --intent "Help me navigate to a shirt that has this on it." \
  --result_dir demo_test_amazon \
  --model gpt-4-vision-preview \
  --action_set_tag som  --observation_type image_som \
  --render
```

This tasks the agent to find a shirt that looks like the provided image (the "This is fine" dog) from Amazon. Have fun!


## Related Repositories 
* [BrowserGym](https://github.com/ServiceNow/BrowserGym): a gym environment for web task automation in the Chromium browser. It supports the evaluation of WebArena and VisualWebArena. The repository features robust web observation processing and supports multiple web-based task benchmarks. 
* [OpenHands](https://github.com/All-Hands-AI/OpenHands): a platform for software development agents powered by AI. It supports the evaluation of WebArena. More details can be found [here](https://github.com/All-Hands-AI/OpenHands/blob/5100d12cea2cd35c30a22e25fbac376b72ed0981/evaluation/webarena/README.md?plain=1)

## Citation
If you find our environment or our models useful, please consider citing <a href="https://jykoh.com/vwa" target="_blank">VisualWebArena</a> as well as <a href="https://webarena.dev/" target="_blank">WebArena</a>:
```
@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv preprint arXiv:2401.13649},
  year={2024}
}

@article{zhou2024webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={ICLR},
  year={2024}
}
```