# EyeQWidget

## Overview
EyeQWidget is a GUI application for annotation, visualization, and quantitative analysis of multifocal `.czi` images of fish and fish eyes in different channels. It provides an intuitive interface for handling `.czi` files, automating analysis, and saving results in structured formats.

## Features
- Supports `.czi` image format.
- GUI-based interaction (no command-line arguments required).
- Saves analysis results and intermediate data in structured files.
- Automatic and manual annotation tools.
- Multi-channel visualization.

## Installation
### Requirements
Ensure you have Python >= 3.9 installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

### Running the Application
Simply double-click the script to run the GUI. No command-line arguments are required.

## Usage
### Setup and Data Organization
1. Organize Your Data. Before you start processing, ensure your data is organized as follows:

#### Naming Scheme:
Use the naming convention: **group_age_zoom_part_id_comment_collector.czi**. The name should include information about group, age, zoom, part, and specimen ID. The part field should be empty for the first image (e.g., part_ _COLLECTOR.czi). The widget accepts X2 (whole fish), X10 (whole eye) and X20 (further zoom on eye)

#### Data Storage:
All images should be stored in a single folder without any additional data files.

#### Imaging Orientation:

Follow the correct imaging orientation and channel order as per the system specifications. X10 and X2 (or X2.5) images contain brightfield, and X20 contain reflectance.
Make sure all X2 files are similarly oriented. they may be either way and specified to the widget, but they should be consistent.

### Making the Folders and Data Tables
2. Create Image-per-Row Table. 
- Click the "Make Image/Stack-per-Row Table" button. This will create a table (tab.csv) summarizing the images and their attributes (group, age, zoom, etc.).
- Check the table (tab.csv) for correctness (e.g., no invalid entries in the age column). If errors are found, modify the file names and repeat steps 1 and 2.

3. Create Single Images Folder. 
- Click the "Make Single Image Folder" button and select the tab.csv file. This will generate individual images and save them in the singles folder. This step may take a few minutes.

4. Create Sample-per-Row Table.
- Click the "Make Sample-per-Row Table" button and select the singles folder. This creates the stab.csv file.

5. View Sample-per-Row Results.
- Click the "Show Sample-per-Row" button and select the stab.csv file.
- Review the sample-per-row data for accuracy (e.g., file name correctness, no empty slots). If any issues are found, correct them and repeat steps 4 and 5.

6. Automate Masking. 
- Click the "Automate Masks" button and select the stab.csv file. 
- Specify the orientation of the X2 files. This generates a stabi.npy file (containing image data) and a masks folder within the singles folder with the automated results.
*Note: that the widget always saves a stabi.npy file (even after simple operations) and overwrites old files, so if you want to "checkpoint" do this by changing the name of the file.*

7. Load Data for Further Use. 
- Click "Load Data" and select the stabi.npy file.
*Always use the widget to interact with files to avoid errors. Do not manually modify files in the folder.*

8. Saving Data
The tab.csv, stab.csv, and stabi.npy files will automatically save with these exact names.

### Editing the Automated Results
Once the data is processed, you can edit the automated results using the widget. Here’s how to handle different tasks:

#### Stitching Images
- Click a stiched image.
- A paint window will open with two images for stitching.
 -Move the images to align them, and remove excess areas (e.g., white regions).
- Save the image (use Ctrl + S) and click OK.
- Regret Action: Click Cancel to discard changes.
- The edited image will appear in the table with a bounding circle around the fish.

#### Fish Length and Eyelength Measurement
- Click a length measurement image.
- A paint window will open with the image for measurement.
- Draw a straight white line to represent the length of the fish or eye.
- Save the image (use Ctrl + S) and click OK.
- Regret Action: Click Cancel to discard changes.
- The program will display a bounding circle around the measured line.

#### Reflectance Masking
- Click a masking X20 image.
- Draw a solid white mask over the area to be analyzed (e.g., the entire eye). Save and click OK.
- Draw a second mask over the areas to be excluded (e.g., the iris). Save and click OK.
- *Both steps must be completed, even if only the first step is needed.*
- Use the 20X Threshold Button if you want to apply a pre-configured threshold suitable for 20X magnification images when painting.

### Getting and Analyzing Results
9. Thresholding - Click the "Show Thresholded" button and enter a threshold value (0-255). to view fluorescence 20X images, use a negative value. The program will display thresholded images for you to inspect.

10. Generating Results
- After setting the threshold, click "Get Results" and select the stabi.npy file.
- Enter your thresholds again to generate the results.
- The program will create a rtab.csv file containing the analysis results (e.g., measurements based on the threshold).

### Troubleshooting and Notes
Common Errors
- Missing Files: Ensure all required images are in the correct folder.
- Thresholding Issues: Adjust the threshold to ensure accurate segmentation.
- Missing/Faulty Automatic Results: The automatic steps may fail, this is why manual annotation is also available. If you get many problems with your data, make sure it is all valid data. if even so the widget fails, further investigation may be needed.

### Important Notes
**Always use the widget to manage your files. Do not manually modify files in the folder to avoid corrupting the data.**

The stabi.npy file contains the image data and can be shared independently of the original images (this is not quite perfect).


## License
MIT License.

