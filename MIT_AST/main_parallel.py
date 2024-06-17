import concurrent.futures
from tqdm import tqdm
import os
import logging
from pipeline_def_ast import run_mit_ast_pipeline 
import pandas as pd

# Define the paths and labels
garden_folder1 = "/Users/evgenynazarenko/DACS_3_year/Thesis/GardenFiles23/garden_25112023"
garden_folder2 = "/Users/evgenynazarenko/DACS_3_year/Thesis/GardenFiles23/garden_30102023"

# Define the absolute path to the Excel file
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, 'MIT_AST_label_map.xlsx')

labels_df = pd.read_excel(excel_path)
human_labels = labels_df[labels_df['source']=='human']
print(human_labels)


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_mit_ast_pipeline, (garden_folder1, human_labels)),
            executor.submit(run_mit_ast_pipeline, (garden_folder2, human_labels))
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                logging.error(f"Error occurred: {e}")

# Ensure this block is executed only if the script is run directly, not when imported
if __name__ == '__main__':
    main()