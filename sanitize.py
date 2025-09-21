import onnx
import sys
import os

def convert_and_save_model(input_path, output_basename):
    """
    Loads an ONNX model and its data, then saves it as a new, clean pair
    with the specified output basename and a correct internal reference.

    Args:
        input_path (str): The path to the source .onnx model.
        output_basename (str): The desired full path for the output files,
                               without the .onnx or .onnx_data extension.
                               e.g., 'output/my_model'
    """
    # 1. Define the final output filenames
    output_model_path = output_basename + '.onnx'
    # The 'location' field inside the ONNX file should be just the filename, not the full path.
    output_data_filename = os.path.basename(output_basename) + '.onnx_data'

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 2. Load the source model and its external data into memory
    try:
        print(f"Loading model from '{input_path}'...")
        # load_external_data=True finds the .data file by convention and loads it.
        model = onnx.load(input_path, load_external_data=True)
        print("Model and data loaded successfully.")

    except Exception as e:
        print(f"\nError loading the model. Ensure its .data file is in the same directory.")
        print(f"Original error: {e}")
        return

    # 3. Save the model to the new paths with the correct internal reference
    try:
        print(f"\nSaving new model to '{output_model_path}'...")
        print(f"Internal reference will be set to '{output_data_filename}'.")
        
        onnx.save(
            model,
            output_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output_data_filename, # This writes the correct string internally
            size_threshold=1024
        )
        print("\nConversion complete!")
        print("You now have a perfectly matched pair of files:")
        print(f"  - Model: {output_model_path}")
        print(f"  - Data:  {output_basename + '.onnx_data'}")

    except Exception as e:
        print(f"\nError saving the new model: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nUsage: python convert_onnx.py <path_to_input.onnx> <path_for_output_model_basename>")
        print("\nExample: python convert_onnx.py original/model_int8.onnx final/model")
        print("This will create 'final/model.onnx' and 'final/model.onnx_data'")
        sys.exit(1)
        
    input_model_path = sys.argv[1]
    output_model_basename = sys.argv[2]
    
    convert_and_save_model(input_model_path, output_model_basename)