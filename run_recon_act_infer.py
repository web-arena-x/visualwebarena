import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
import AWorld.examples.visualwebarena.utils as u
from vwa_tester import VWATester, VWAPathHandler, VWAConfig

def get_default_config(domain, start_index = 0, end_index = 466) -> VWAConfig:
    args = VWAConfig()
    args.action_set_tag = "som" # action type
    args.vwa_code_path = f'{parent_dir}/'
    args.vwa_data_path = f'{u.get_nas()}/gui_dataset/visualwebarena/'

    args.mode = 'mas' # som, vision, mas
    if args.mode == 'vision':
        args.instruction_path = f'{args.vwa_code_path}/agent/prompts/jsons/vision.json'
        args.observation_type = "image"
    elif args.mode == 'som':
        args.instruction_path = f'{args.vwa_code_path}/agent/prompts/jsons/som.json'
        args.observation_type = "image_som"
    elif args.mode == 'mas':
        args.instruction_path = f'{args.vwa_code_path}/agent/prompts/jsons/mas.json'
        args.observation_type = "image_som"

    args.model = os.getenv('LLM_MODEL_NAME')
    args.eval_provider = os.getenv('LLM_MODEL_NAME')
    
    args.caption_model = 'KevinBlip'

    args.domain = domain
    args.print_time = True
    args.output_response = True
    args.render = True
    args.flush = True
    args.save_trace_enabled = False
    args.render_fail_only = False
    args.test_start_idx = start_index
    args.test_end_idx = end_index

    return args

if __name__ == "__main__":
    domain = sys.argv[1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = get_default_config(domain)
    ph = VWAPathHandler(args)
    metrics_files = u.list_files(ph.metrics_path)
    test_config_base_dir = f'{args.vwa_data_path}/config_files/vwa/test_{args.domain}'
    test_config_files = u.list_files(test_config_base_dir)
    test_config_files = sorted(test_config_files, key=lambda x: int(x.split('.')[0]))
    if not args.flush: 
        test_config_files = [a for a in test_config_files if a not in metrics_files]
    test_config_files = [test_config_base_dir + '/' + a for a in test_config_files]
    args.test_config_files = test_config_files
    tester = VWATester(args)
    tester.test()