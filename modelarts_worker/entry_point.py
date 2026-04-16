import argparse
import json
import os
import moxing as mox

import mindspore_config
from mindspore_config import get_logger
from logic.DataAnalyzer import DataAnalyzer

logger = get_logger("entry_point")

def main():
    parser = argparse.ArgumentParser(description="MiniQDs Optimization Worker")
    parser.add_argument('--data_url', type=str, required=True, 
                        help='OBS path of the JSON file containing input parameters')
    parser.add_argument('--train_url', type=str, required=True, 
                        help='OBS path where the results JSON will be saved')
    args = parser.parse_args()

    logger.info("Starting job on ModelArts...")
    logger.info(f"Input OBS (data_url): {args.data_url}")
    logger.info(f"Output OBS (train_url): {args.train_url}")

    local_input_path = '/cache/input_params.json'
    local_output_path = '/cache/output_results.json'

    try:
        logger.info("Downloading parameters from OBS...")
        mox.file.copy(args.data_url, local_input_path)
        
        with open(local_input_path, 'r', encoding='utf-8') as f:
            request_dict = json.load(f)
            
        logger.info(f"Loaded parameters: {request_dict.get('materials')}, Pop: {request_dict.get('population_size')}")

        csv_path = os.path.join(os.path.dirname(__file__), "materials.csv")
        kappa = request_dict.get("kappa", 0.5)
        
        logger.info("Instantiating DataAnalyzer and running simulation...")
        analyzer = DataAnalyzer(csv_path=csv_path, kappa=kappa)
        result_dict = analyzer.analyze(request_dict)
        
        logger.info("Simulation completed. Saving results locally...")
        result_dict["status"] = "COMPLETED" 
        
        with open(local_output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f)
            
        logger.info("Uploading results to OBS...")
        mox.file.copy(local_output_path, args.train_url)
        
        logger.info("Job finished successfully!")

    except Exception as e:
        logger.exception("Critical error during Worker execution:")
        
        error_dict = {
            "status": "FAILED",
            "error_detail": str(e)
        }
        with open(local_output_path, 'w', encoding='utf-8') as f:
            json.dump(error_dict, f)
        
        if mox.file.exists(os.path.dirname(args.train_url)):
            mox.file.copy(local_output_path, args.train_url)
        raise e 

if __name__ == '__main__':
    main()