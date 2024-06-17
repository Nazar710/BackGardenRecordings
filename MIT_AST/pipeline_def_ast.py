
from mit_ast_prob import MIT_AST_model_prob
import os
from tqdm import tqdm
import pandas as pd


def split_prob_results(prob_results):
    """
    Split the results of the classification into a list of tuples with file path and label and a list of tuples with file path and dictionary
    :param prob_results: list of tuples with file path and classification results dictionary with top labels and their probabilities
                        results = [(file_path, {label: prob, ...}), ...]
    :return: list of tuples with file path and label and a list of tuples with file path and dictionary
    """
    labels = []
    dictionaries = []
    for file_path, label_prob_dict in prob_results:
        # Get the top label
        top_label = max(label_prob_dict, key=label_prob_dict.get)
        labels.append((file_path, top_label))
        dictionaries.append((file_path, label_prob_dict))
    return labels, dictionaries

# human_detected function. finds if human related albels are among the results
def human_detected(prob_results, human_labels):
    """
    Check if any of the labels in the results is a human label and return a list of tuples with file path and a flag
    :param prob_results: list of tuples with file path and classification results dictionary with top 5 labels and their probabilities
                        results = [(file_path, {label: prob, ...}), ...]
    :param human_labels: list of human labels
    :return: list of tuples with file path and 1 if a human label is detected, 0 otherwise
    """
    detected_results = []
    for file_path, label_prob_dict in prob_results:
        # Check if any label is in human_labels
        human_detected = any(label in human_labels for label in label_prob_dict)
        detected_results.append((file_path, 1 if human_detected else 0))
    return detected_results

    # add_labels function. Adds a label to the results if it is a human label
def add_labels(metadata_file_path,results,column_name):
    """
    Add MIT_AST labels to the metadata file
    :param metadata_file_path: path to the metadata file
    :param results: list of tuples with file path and top label 
    :param column_name: name of the column to add to the metadata file            
    :return: path to the new metadata file
    """
    # Create a DataFrame from results
    labels_df = pd.DataFrame(results, columns=['filepath', column_name])
    
    # Extract basenames for comparison
    labels_df['filename'] = labels_df['filepath'].apply(os.path.basename)
    
    # Load the existing metadata DataFrame
    df = pd.read_excel(metadata_file_path)
    
    # Ensure filenames are basenames for comparison
    df['filename'] = df['filename'].apply(os.path.basename)
    
    # Check if the 'BirdNET_label' column exists; if not, create it with NaN values
    if column_name not in df.columns:
        df[column_name] = pd.NA
    
    # Merge the existing df with the labels_df
    df = df.merge(labels_df[['filename', column_name]], on='filename', how='left', suffixes=('', '_new'))
    
    # Update the 'MIT_AST_label' column with new values where available
    df[column_name] = df[column_name+'_new'].combine_first(df[column_name])
    
    # Drop the temporary 'BirdNET_label_new' column
    df.drop(column_name+'_new', axis=1, inplace=True)

    # Construct the new filename
    original_filename = os.path.basename(metadata_file_path)
    new_filename = original_filename.replace('metadata', column_name + '_metadata')
    new_filepath = os.path.join(os.path.dirname(metadata_file_path), new_filename)
    # Save the updated DataFrame to a new Excel file
    df.to_excel(new_filepath, index=False)
    # Remove the original file
    os.remove(metadata_file_path)
    return new_filepath

def find_files(root_folder):
    """
    Finds the metadata file and audio files in the specified root folder.
    The root folder is assumed to contain subfolders with audio files and metadata file.
    The metadata file is assumed to be named '...metadata.xlsx' and the audio files are assumed to have the extension '.wav'.
    parameters:
    - root_folder: The path to the root folder containing the audio files and metadata file.
    returns:
    - metadata_filepath: The path to the metadata file.
    - audio_files: A list of relative paths to the audio files.

    """
    audio_files = []
    metadata_filepath = None
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('metadata.xlsx'):
                metadata_filepath = os.path.join(dirpath, filename)
                print(f"Found metadata file: {metadata_filepath}")

            elif filename.endswith('.wav'):
                audio_files.append(os.path.join(dirpath, filename))
            

    return metadata_filepath, audio_files



def classify_files_ast_prob(filepaths_list):
    """
    Classify a list of audio files using the MIT AST model
    :param filepaths_list: list of file paths
    :return: list of tuples with file path and classification results a list of tuples for each file
            where each tuple contains the file path and a results tuple.
            res tuple contains winning label + dictionary with top 5 labels and their probabilities
            results = [(file_path, (label,{label:prob,...})), ...]
    """
    results = []
    model_prob = MIT_AST_model_prob()
    for file_path in tqdm(filepaths_list, desc="Classifying files"):
        res = model_prob.classify(file_path)
        try:
            if len(res) > 0:
                results.append((file_path, res))
        except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return results
        
      
    

# mit_ast_pipeline
def mit_ast_pipeline(folder_path,human_labels_list):
    """
    Classify all audio files in the folder using the MIT AST model and add the human labels to the metadata file
    :param folder_path: path to the folder with audio files
    :param human_labels_list: list of human labels from MIT_AST_label_map.xlsx
    :return: None
    """
    # Get a list of all audio files in the folder
    metadata_filepath, audio_files = find_files(folder_path)
    
    # Classify the audio files using the MIT AST model
    results = classify_files_ast_prob(audio_files)

    # Split the results into labels and dictionaries
    labels, dictionaries = split_prob_results(results)
    
    # add labels to the metadata file
    new_metadata_filepath = add_labels(metadata_filepath,labels,'MIT_AST')
    print(metadata_filepath)

    # Check if any of the labels in the results is a human label
    human_detected_results = human_detected(dictionaries, human_labels_list)

    # add human detected labels to the metadata file
    add_labels(new_metadata_filepath,human_detected_results,'Human_detected')

def run_mit_ast_pipeline(args):
    folder_path, human_labels_list = args
    mit_ast_pipeline(folder_path, human_labels_list)
    return f"Completed processing for folder: {folder_path}"