import os

def trim_numbers_from_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                trimmed_line = ' '.join(line.split()[:-8]) + '\n'
                file.write(trimmed_line)
    except UnicodeDecodeError as e:
        print(f"Error decoding file {file_path}: {e}")
        return False
    return True

directory = 'E:/BaiduNetdiskDownload/detect_plate_datasets/val_detect/val_detect/shandouble_1'   # Change this to your directory

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        print(f"Processing {filename}...")
        if trim_numbers_from_lines(file_path):
            print(f"Finished processing {filename}.")
        else:
            print(f"Skipped {filename} due to an error.")

print("All files have been processed.")
