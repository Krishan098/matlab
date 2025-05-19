Experiment 1: Image Processing Concept

% Experiment 1: Image Processing Concept
I = imread('pout.tif');
J = imcomplement(I);

figure, imshow(I);
title('Original Image');

figure, imshow(J);
title('Complement Image');

K = imadjust(I, [0, 0.4], [0.5, 1]);

figure, imshow(K);
title('Adjusted Image');

Experiment 2: Histogram Equalization

% Experiment 2: Histogram Equalization
I = imread('trees.tif');
J = imcomplement(I);

figure, imshow(I);
title('Original Image');

figure, imshow(J);
title('Complement Image');

I_eq = histeq(I);

figure, imhist(I_eq, 64);
title('Equalized Histogram');

n = numel(I);
p = imhist(I) / n;

figure, plot(p);
title('Normalized Histogram');

K = imadjust(I, [0 1], [0.4 1], 0.5);
figure, imshow(K);
title('Adjusted Image');

T = maketform('affine', [0.3 0 0; 0.5 1 0; 0 0 1]);
I2 = imtransform(I, T);
figure, imshow(I2);
title('Affine Transformed Image');

Experiment 3: Averaging Filter

% Experiment 3: Averaging Filter in Spatial Domain
I = imread('trees.tif');

subplot(2, 2, 1);
imshow(I);
title('Original Image');

f = ones(3, 3) / 9;
h = imfilter(I, f, 'circular');

subplot(2, 2, 2);
imshow(h);
title('Averaged Image');

Experiment 4: Opening and Closing

% Experiment 4: Morphological Opening and Closing
f = imread('coins.png');
se = strel('square', 20);

fo = imopen(f, se);
fc = imclose(f, se);
foc = imclose(fo, se);

figure, imshow(f);
title('Input Image');

figure, imshow(fo);
title('Opened Image');

figure, imshow(fc);
title('Closed Image');

figure, imshow(foc);
title('Closed After Opened Image');

Experiment 5: Region of Interest (Buggy Example)

This experiment contains an incorrect command roi ill. Here's a corrected concept using placeholder action:

% Experiment 5: Region of Interest
clc;
close all;

load trees;
I = ind2gray(X, map);
imshow(I);
title('Original Image');

% Replace 'roi ill' with an example fill/processing:
roi_mask = roipoly();
I2 = I;
I2(roi_mask) = 255;  % Fill ROI with white

figure, imshow(I2);
title('Output Image with ROI Filled');

Experiment 6: Edge Detection

% Experiment 6: Edge Detection Algorithms
I = imread('coins.png');
figure, imshow(I);
title('Original Image');

h = ones(5,5) / 25;
b = imfilter(I, h);

figure, imshow(b);
title('Filtered Image');

c = edge(b, 'sobel');
figure, imshow(c);
title('Sobel Edge');

d = edge(b, 'prewitt');
figure, imshow(d);
title('Prewitt Edge');

e = edge(b, 'roberts');
figure, imshow(e);
title('Roberts Edge');

f = edge(b, 'canny');
figure, imshow(f);
title('Canny Edge');

Experiment 7: Sharpening using Gradient Mask

% Experiment 7: Sharpening using Sobel Gradient
I = imread('coins.png');

subplot(2,2,1);
imshow(I);
title('Original Image');

h = fspecial('sobel');
F = imfilter(I, h, 'replicate');

subplot(2,2,2);
imshow(F);
title('Filtered Image by Sobel Mask');

s = imadd(I, F);  % Sharpening using addition
subplot(2,2,4);
imshow(s);
title('Final A/P Image');

Experiment 9: DCT/IDCT Visualization

% Experiment 9: DCT Basis Matrix Visualization
clc; clear; close all;

m = input('Enter the basis matrix dimension: ');
n = m;  % Assuming square matrix

alpha1 = ones(1, m) * sqrt(2 / m);
alpha1(1) = sqrt(1 / m);
alpha2 = ones(1, n) * sqrt(2 / n);
alpha2(1) = sqrt(1 / n);

a = zeros(m, n, m, n);

for u = 0:m-1
    for v = 0:n-1
        for x = 0:m-1
            for y = 0:n-1
                a(u+1,v+1,x+1,y+1) = ...
                    alpha1(u+1) * alpha2(v+1) * ...
                    cos((2*x+1)*u*pi/(2*m)) * ...
                    cos((2*y+1)*v*pi/(2*n));
            end
        end
    end
end

figure;
k = 1;
for i = 1:m
    for j = 1:n
        subplot(m, n, k);
        imagesc(squeeze(a(i,j,:,:))); colormap(gray); axis off;
        k = k + 1;
    end
end

Experiment 9 (Duplicate): ROI Fill with roipoly

% Experiment 9 (alt): ROI Fill using roipoly
clc; close all;

load trees;
imshow(X, map);
title('Original Image');

fprintf('Select the region of interest using roipoly.\n');
roi_mask = roipoly();

filled_image = X;
filled_image(roi_mask) = 255;

figure;
subplot(1,2,1);
imshow(X, map);
title('Original Image');

subplot(1,2,2);
imshow(filled_image, map);
title('Image with Filled Region of Interest');

Let me know if you'd like these saved in .m files or combined into a script.
You said:
in exp5 the image output is not shown until i close the original image figure
ChatGPT said:

You're right — in Experiment 5, the roipoly() function is blocking because it launches an interactive selection tool tied to the current figure, and MATLAB pauses execution until the user completes the ROI selection. Therefore, all code after roipoly() doesn't execute until you complete the ROI selection, not necessarily when you close the figure, but when you finish drawing the ROI (usually by double-clicking).

However, to improve the behavior and avoid confusion, it's best to:

    Show the original image clearly before asking for ROI.

    Use gcf to keep the figure open properly.

    Display the final image in a new figure after selection.

Here’s the improved and clearer version of Experiment 5:

% Experiment 5: Region of Interest Fill
clc;
close all;

% Load and display the original image
load trees;
I = ind2gray(X, map);

hFig = figure; % Ensure it's assigned to a variable
imshow(I);
title('Original Image');
fprintf('Use the mouse to select a polygonal ROI, then double-click to finish.\n');

% Get the ROI mask interactively
roi_mask = roipoly(hFig);  % Will wait until selection is finished

% Process the ROI
I2 = I; 
I2(roi_mask) = 255;

% Show the output image
figure;
imshow(I2);
title('Output Image with ROI Filled');

Notes:

    roipoly(hFig) ties the interaction to the specific figure displaying the image.

    Execution pauses until you double-click or close the selection — this is expected behavior.

    A new figure is used to show the result so you don’t overwrite the interactive one.

Let me know if you'd prefer a non-interactive method using a predefined ROI instead.
You said:
now give explanation about every exp, of the code as well as what the theory explanation in depth.
ChatGPT said:

Absolutely. Here is a detailed breakdown for each experiment including:

    In-depth theoretical explanation of the image processing concept.

    Explanation of each line of the corrected MATLAB code.

🧪 Experiment 1: Image Processing Concept – Image Complement & Adjustment
📘 Theory:

    Digital Image Processing (DIP) involves manipulating digital images using a computer algorithm.

    A complement (negative) of an image is calculated by subtracting each pixel value from the maximum intensity (usually 255 for 8-bit images). This is useful for analyzing details in darker regions.

    imadjust is used to enhance contrast by remapping intensity values from a specified range.

🧠 Code Explanation:

I = imread('pout.tif');        % Load a grayscale image into variable I
J = imcomplement(I);          % Compute the complement (negative) of the image
figure, imshow(I);            % Display original image
title('Original Image');

figure, imshow(J);            % Display complement image
title('Complement Image');

K = imadjust(I, [0, 0.4], [0.5, 1]);  % Adjust contrast: map [0,0.4] → [0.5,1]
figure, imshow(K);            % Show the contrast-enhanced image
title('Adjusted Image');

🧪 Experiment 2: Histogram Equalization
📘 Theory:

    A histogram shows pixel intensity distribution.

    Histogram Equalization enhances contrast by redistributing pixel intensities more uniformly.

    Useful for images with poor contrast or lighting.

🧠 Code Explanation:

I = imread('trees.tif');       % Read the grayscale image
J = imcomplement(I);           % Get the complement of the image

figure, imshow(I);
title('Original Image');

figure, imshow(J);
title('Complement Image');

I_eq = histeq(I);              % Perform histogram equalization
figure, imhist(I_eq, 64);
title('Equalized Histogram');

n = numel(I);                  % Total number of pixels
p = imhist(I) / n;             % Normalized histogram

figure, plot(p);
title('Normalized Histogram');

K = imadjust(I, [0 1], [0.4 1], 0.5); % Contrast stretch with gamma
figure, imshow(K);
title('Adjusted Image');

T = maketform('affine', [0.3 0 0; 0.5 1 0; 0 0 1]); % Affine transformation matrix
I2 = imtransform(I, T);       % Apply affine transformation
figure, imshow(I2);
title('Affine Transformed Image');

🧪 Experiment 3: Averaging Filter (Smoothing)
📘 Theory:

    Smoothing filters reduce image noise.

    Averaging filter replaces each pixel with the average of its neighbors.

    Used in pre-processing to reduce sharp transitions (noise).

🧠 Code Explanation:

I = imread('trees.tif');       % Load the original image

subplot(2,2,1);
imshow(I);
title('Original Image');

f = ones(3, 3) / 9;            % Define a 3x3 averaging kernel
h = imfilter(I, f, 'circular');% Apply filter with circular padding

subplot(2,2,2);
imshow(h);
title('Averaged Image');

🧪 Experiment 4: Morphological Opening and Closing
📘 Theory:

    Opening = erosion followed by dilation: removes small foreground objects.

    Closing = dilation followed by erosion: fills small holes.

    Useful in noise removal, object segmentation, and shape analysis.

🧠 Code Explanation:

f = imread('coins.png');            % Read binary or grayscale image
se = strel('square', 20);           % Create square structuring element

fo = imopen(f, se);                 % Apply opening operation
fc = imclose(f, se);                % Apply closing operation
foc = imclose(fo, se);              % Apply closing on opened image

figure, imshow(f);
title('Input Image');

figure, imshow(fo);
title('Opened Image');

figure, imshow(fc);
title('Closed Image');

figure, imshow(foc);
title('Closed After Opened Image');

🧪 Experiment 5: Region of Interest (ROI) Fill
📘 Theory:

    ROI (Region of Interest): a user-defined subset of an image for localized processing.

    roipoly() lets you interactively select polygonal regions.

    Useful for segmenting, masking, or measuring specific areas.

🧠 Code Explanation:

clc; close all;

load trees;
I = ind2gray(X, map);         % Convert indexed image to grayscale

hFig = figure;
imshow(I);
title('Original Image');

roi_mask = roipoly(hFig);     % User selects ROI interactively

I2 = I;
I2(roi_mask) = 255;           % Fill the ROI area with white (255)

figure, imshow(I2);
title('Output Image with ROI Filled');

🧪 Experiment 6: Edge Detection
📘 Theory:

    Edges are rapid intensity changes—key for detecting object boundaries.

    Common operators:

        Sobel: horizontal/vertical edges.

        Prewitt: similar to Sobel but simpler.

        Roberts: diagonal edges.

        Canny: robust and multi-stage for clean detection.

🧠 Code Explanation:

I = imread('coins.png');      % Load grayscale image
figure, imshow(I);
title('Original Image');

h = ones(5,5) / 25;           % Smoothing filter
b = imfilter(I, h);           % Apply smoothing

c = edge(b, 'sobel');         % Detect edges using Sobel
figure, imshow(c);
title('Sobel Edge');

d = edge(b, 'prewitt');
figure, imshow(d);
title('Prewitt Edge');

e = edge(b, 'roberts');
figure, imshow(e);
title('Roberts Edge');

f = edge(b, 'canny');
figure, imshow(f);
title('Canny Edge');

🧪 Experiment 7: Sharpening using Gradient Mask
📘 Theory:

    Gradient mask (Sobel) detects edges or transitions.

    Sharpening emphasizes transitions using edge-based boosting.

    Adds the gradient back to the original to highlight edges.

🧠 Code Explanation:

I = imread('coins.png');

subplot(2,2,1);
imshow(I);
title('Original Image');

h = fspecial('sobel');        % Sobel kernel
F = imfilter(I, h, 'replicate');  % Apply filter

subplot(2,2,2);
imshow(F);
title('Filtered Image by Sobel Mask');

s = imadd(I, F);              % Add filtered result to original for sharpening

subplot(2,2,4);
imshow(s);
title('Final A/P Image');

🧪 Experiment 9: DCT/IDCT Visualization
📘 Theory:

    Discrete Cosine Transform (DCT) converts spatial data into frequency components.

    Used in JPEG compression and feature extraction.

    DCT basis functions are cosines at different frequencies.

    IDCT (Inverse DCT) reconstructs the image from frequency data.

🧠 Code Explanation:

clc; clear; close all;

m = input('Enter the basis matrix dimension: ');
n = m;

alpha1 = ones(1, m) * sqrt(2 / m);
alpha1(1) = sqrt(1 / m);
alpha2 = ones(1, n) * sqrt(2 / n);
alpha2(1) = sqrt(1 / n);

a = zeros(m, n, m, n); % 4D matrix to store basis

for u = 0:m-1
    for v = 0:n-1
        for x = 0:m-1
            for y = 0:n-1
                a(u+1,v+1,x+1,y+1) = ...
                    alpha1(u+1) * alpha2(v+1) * ...
                    cos((2*x+1)*u*pi/(2*m)) * ...
                    cos((2*y+1)*v*pi/(2*n));
            end
        end
    end
end

figure;
k = 1;
for i = 1:m
    for j = 1:n
        subplot(m, n, k);
        imagesc(squeeze(a(i,j,:,:)));
        colormap(gray);
        axis off;
        k = k + 1;
    end
end

Would you like this formatted into a printable PDF or .m files for each experiment?
You said:
make the theory explanation more in depth, and compile into a pdf
ChatGPT said:

Always show details

from fpdf import FPDF

# Create a PDF document
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Digital Image Processing Experiments - In-depth Theory & Code (MATLAB R2015a)", ln=True, align='C')
pdf.ln(10)
pdf.set_font("Arial", '', 12)

# Helper function to add sections
def add_experiment(title, theory, code):
    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"{title}", align='L')
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 8, "Theory:")
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, theory)
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 8, "MATLAB Code:")
    pdf.set_font("Courier", '', 10)
    for line in code.strip().split('\n'):
        pdf.multi_cell(0, 6, line)
    pdf.ln(5)

# Detailed theoretical explanations and cleaned code
experiments = [
    {
        "title": "Experiment 1: Image Complement and Contrast Adjustment",
        "theory": (
            "This experiment introduces fundamental image processing operations such as obtaining the complement (negative) "
            "and enhancing contrast using intensity mapping. The complement of an image is formed by subtracting each pixel value "
            "from the maximum possible value (255 for 8-bit images). This inversion is useful for enhancing details in darker regions. "
            "Contrast enhancement via `imadjust` maps a specific input intensity range to a new output range, making dark regions lighter "
            "or bright regions darker, thus improving visibility. This is often used in preprocessing pipelines before applying more "
            "advanced algorithms like segmentation or feature extraction."
        ),
        "code": """
I = imread('pout.tif');
J = imcomplement(I);
figure, imshow(I);
title('Original Image');
figure, imshow(J);
title('Complement Image');
K = imadjust(I, [0, 0.4], [0.5, 1]);
figure, imshow(K);
title('Adjusted Image');
"""
    },
    {
        "title": "Experiment 2: Histogram Equalization",
        "theory": (
            "Histogram Equalization enhances image contrast by redistributing the pixel intensity values. In a poorly contrasted image, "
            "the histogram tends to be clustered around certain intensity values. This technique spreads out the most frequent intensity "
            "values, effectively enhancing contrast in areas with low dynamic range. It is especially useful in medical imaging, satellite "
            "imagery, and any application where visual clarity is crucial. `histeq` automatically equalizes the histogram. Normalizing the "
            "histogram (by dividing pixel counts by the total number of pixels) helps visualize relative frequencies and is essential in many "
            "statistical image analysis techniques."
        ),
        "code": """
I = imread('trees.tif');
J = imcomplement(I);
figure, imshow(I);
title('Original Image');
figure, imshow(J);
title('Complement Image');
I_eq = histeq(I);
figure, imhist(I_eq, 64);
title('Equalized Histogram');
n = numel(I);
p = imhist(I) / n;
figure, plot(p);
title('Normalized Histogram');
K = imadjust(I, [0 1], [0.4 1], 0.5);
figure, imshow(K);
title('Adjusted Image');
T = maketform('affine', [0.3 0 0; 0.5 1 0; 0 0 1]);
I2 = imtransform(I, T);
figure, imshow(I2);
title('Affine Transformed Image');
"""
    }
]

# Add each experiment to the PDF
for exp in experiments:
    add_experiment(exp["title"], exp["theory"], exp["code"])

# Save the PDF
pdf_output_path = "/mnt/data/DIP_Experiments_Theory_and_Code_Part1.pdf"
pdf.output(pdf_output_path)

Result

''

The first part of the PDF has been created with in-depth theory and code explanations for:

    Experiment 1: Image Complement and Contrast Adjustment

    Experiment 2: Histogram Equalization

You can download it here:
📄 DIP_Experiments_Theory_and_Code_Part1.pdf

Would you like me to continue compiling the rest of the experiments into the same or a new PDF?
You said:
same pdf
ChatGPT said:

Always show details

# Continue adding the remaining experiments to the same PDF
experiments_remaining = [
    {
        "title": "Experiment 3: Averaging Filter (Smoothing)",
        "theory": (
            "Smoothing is a fundamental image processing technique used to reduce noise and minor variations in an image. "
            "The averaging filter (also known as a box filter) calculates the average value of neighboring pixels in a window (e.g., 3x3) "
            "and assigns it to the central pixel. This process blurs the image slightly but makes it more robust to noise. "
            "Circular padding ensures that the image borders are handled gracefully by wrapping around the image data, avoiding edge artifacts."
        ),
        "code": """
I = imread('trees.tif');
subplot(2,2,1);
imshow(I);
title('Original Image');
f = ones(3, 3) / 9;
h = imfilter(I, f, 'circular');
subplot(2,2,2);
imshow(h);
title('Averaged Image');
"""
    },
    {
        "title": "Experiment 4: Morphological Opening and Closing",
        "theory": (
            "Morphological operations are key tools in binary and grayscale image analysis. "
            "Opening (erosion followed by dilation) is used to remove small foreground objects, while closing (dilation followed by erosion) "
            "is used to fill small holes in the foreground. These operations rely on a structuring element, which defines the shape and size "
            "of the neighborhood used for the operation. They're especially useful for pre- or post-processing steps in object detection and segmentation tasks."
        ),
        "code": """
f = imread('coins.png');
se = strel('square', 20);
fo = imopen(f, se);
fc = imclose(f, se);
foc = imclose(fo, se);
figure, imshow(f);
title('Input Image');
figure, imshow(fo);
title('Opened Image');
figure, imshow(fc);
title('Closed Image');
figure, imshow(foc);
title('Closed After Opened Image');
"""
    },
    {
        "title": "Experiment 5: Region of Interest (ROI) Fill",
        "theory": (
            "A Region of Interest (ROI) allows selective processing of an image segment defined by the user. "
            "`roipoly` enables interactive polygon selection, which returns a binary mask indicating the region's area. "
            "Filling the ROI is useful for tasks like object segmentation, occlusion removal, or focus analysis. "
            "This experiment demonstrates interactive selection and localized manipulation of the image matrix using ROI masks."
        ),
        "code": """
clc; close all;
load trees;
I = ind2gray(X, map);
hFig = figure;
imshow(I);
title('Original Image');
roi_mask = roipoly(hFig);
I2 = I;
I2(roi_mask) = 255;
figure, imshow(I2);
title('Output Image with ROI Filled');
"""
    },
    {
        "title": "Experiment 6: Edge Detection",
        "theory": (
            "Edge detection is critical for identifying boundaries in images. It reveals object outlines by detecting significant intensity changes. "
            "Multiple operators exist:\n"
            "- Sobel: emphasizes vertical and horizontal edges.\n"
            "- Prewitt: similar to Sobel but computationally simpler.\n"
            "- Roberts: sensitive to diagonal edges.\n"
            "- Canny: multi-stage detector, known for its precision and noise suppression.\n"
            "Applying a smoothing filter before edge detection reduces false edges due to noise."
        ),
        "code": """
I = imread('coins.png');
figure, imshow(I);
title('Original Image');
h = ones(5,5) / 25;
b = imfilter(I, h);
c = edge(b, 'sobel');
figure, imshow(c);
title('Sobel Edge');
d = edge(b, 'prewitt');
figure, imshow(d);
title('Prewitt Edge');
e = edge(b, 'roberts');
figure, imshow(e);
title('Roberts Edge');
f = edge(b, 'canny');
figure, imshow(f);
title('Canny Edge');
"""
    },
    {
        "title": "Experiment 7: Sharpening using Gradient Mask",
        "theory": (
            "Image sharpening enhances edges and fine details. It is often used to highlight important structures. "
            "A gradient mask like the Sobel filter detects edges, and adding this gradient back to the original image boosts sharpness. "
            "This technique is crucial in feature detection tasks and improves the visibility of edges that may be lost due to blurring."
        ),
        "code": """
I = imread('coins.png');
subplot(2,2,1);
imshow(I);
title('Original Image');
h = fspecial('sobel');
F = imfilter(I, h, 'replicate');
subplot(2,2,2);
imshow(F);
title('Filtered Image by Sobel Mask');
s = imadd(I, F);
subplot(2,2,4);
imshow(s);
title('Final A/P Image');
"""
    },
    {
        "title": "Experiment 9: DCT/IDCT Visualization",
        "theory": (
            "The Discrete Cosine Transform (DCT) converts spatial domain data into frequency components. It is widely used in image compression (e.g., JPEG). "
            "Lower-frequency components capture the majority of visual information, while higher frequencies can often be discarded. "
            "Visualizing DCT basis functions helps understand how individual frequency elements contribute to image reconstruction. "
            "This experiment manually constructs and visualizes the DCT basis matrix using cosine functions, exploring the mathematical structure behind DCT."
        ),
        "code": """
clc; clear; close all;
m = input('Enter the basis matrix dimension: ');
n = m;
alpha1 = ones(1, m) * sqrt(2 / m);
alpha1(1) = sqrt(1 / m);
alpha2 = ones(1, n) * sqrt(2 / n);
alpha2(1) = sqrt(1 / n);
a = zeros(m, n, m, n);
for u = 0:m-1
    for v = 0:n-1
        for x = 0:m-1
            for y = 0:n-1
                a(u+1,v+1,x+1,y+1) = ...
                    alpha1(u+1) * alpha2(v+1) * ...
                    cos((2*x+1)*u*pi/(2*m)) * ...
                    cos((2*y+1)*v*pi/(2*n));
            end
        end
    end
end
figure;
k = 1;
for i = 1:m
    for j = 1:n
        subplot(m, n, k);
        imagesc(squeeze(a(i,j,:,:)));
        colormap(gray);
        axis off;
        k = k + 1;
    end
end
"""
    }
]

# Add remaining experiments to the same PDF
for exp in experiments_remaining:
    add_experiment(exp["title"], exp["theory"], exp["code"])

# Save the updated PDF
pdf.output(pdf_output_path)

Result

''

✅ The full PDF has been updated with in-depth theoretical explanations and corrected MATLAB R2015a code for all the experiments.

You can download it here:
📄 DIP_Experiments_Theory_and_Code_Part1.pdf

Let me know if you'd like:

    Visual output examples for the experiments,

    A second PDF with sample results,

    Or .m files for each experiment.

You said:
ok just make the theory explanation in more detail (not code explanation) and write in a message instead of pdf
ChatGPT said:

Absolutely. Here’s a more detailed and educational explanation of the theoretical concepts for each Digital Image Processing experiment based on your document. These are suitable for academic understanding or inclusion in lab reports.
🧪 Experiment 1: Image Complement and Contrast Adjustment
🔬 Theory (In-depth):

In digital image processing, every image is composed of pixel intensity values. A grayscale image typically uses pixel values in the range [0, 255], where 0 is black and 255 is white.

    Image Complement (Negative): This operation inverts the intensity values of an image. For an 8-bit grayscale image, the complement is calculated as 255 - pixel_value. This is particularly useful when analyzing features in dark regions or when the subject and background need to be swapped visually (e.g., X-ray imaging).

    Contrast Enhancement: Sometimes, image details are hard to perceive due to poor lighting or low dynamic range. The imadjust() function in MATLAB performs contrast stretching or intensity mapping. It maps a narrow range of pixel intensities (e.g., [0, 0.4]) to a broader one ([0.5, 1]), thereby making features more distinguishable.

This experiment introduces fundamental tools for preprocessing steps used in feature extraction, segmentation, or enhancement.
🧪 Experiment 2: Histogram Equalization
🔬 Theory (In-depth):

Histogram equalization is a method to improve the global contrast of an image. In many real-world scenarios (e.g., underexposed photographs), pixel intensities may occupy a narrow band of values.

    The histogram of an image plots the number of pixels at each intensity level.

    Equalization stretches and redistributes these intensities to make them more uniformly spread, thus enhancing contrast.

This method works particularly well for images with backgrounds and foregrounds that are both bright or both dark. It is heavily used in fields such as medical imaging, thermal imaging, and remote sensing, where visual clarity of subtle features is crucial.

However, for color images or images with already good contrast, histogram equalization might create unnatural effects.
🧪 Experiment 3: Averaging Filter (Smoothing)
🔬 Theory (In-depth):

Smoothing is a low-pass filtering technique used to suppress high-frequency noise.

    The averaging filter or box filter calculates the average of all the pixels in a local neighborhood (like a 3×3 window) and replaces the center pixel with this average.

    This leads to a blurring effect that smooths sharp transitions (like noise), making the image more stable for further processing.

This method is essential in preprocessing stages such as edge detection and object segmentation to prevent false detections due to noise. However, too much smoothing can also remove important image features.
🧪 Experiment 4: Morphological Opening and Closing
🔬 Theory (In-depth):

Morphological operations are used in binary and grayscale image processing to analyze and process geometrical structures.

    Opening is erosion followed by dilation. It removes small objects/noise from an image (e.g., dust, scratches).

    Closing is dilation followed by erosion. It fills small holes or gaps in objects.

Both operations rely on a structuring element (SE) (e.g., a square or disk), which defines the neighborhood size and shape.

Morphological techniques are widely used in object detection, character recognition, and medical imaging, especially when the shape or boundary of objects is of interest.
🧪 Experiment 5: Region of Interest (ROI) Fill
🔬 Theory (In-depth):

An ROI (Region of Interest) is a specific part of an image selected for focused processing or analysis.

    ROIs are useful when the entire image does not require processing, only specific areas—like a tumor in a scan or a person’s face in a photograph.

    The ROI is often user-defined using polygon selection tools (e.g., roipoly()), resulting in a binary mask.

    The selected region can be analyzed, enhanced, or filled (e.g., for object removal, occlusion filling, or segmentation training).

ROIs are integral in medical imaging, object tracking, image annotation, and pattern recognition.
🧪 Experiment 6: Edge Detection
🔬 Theory (In-depth):

Edges represent object boundaries where intensity changes abruptly. Edge detection is one of the most critical operations in image analysis.

Several operators are used:

    Sobel: Computes gradient magnitude in both vertical and horizontal directions.

    Prewitt: Similar to Sobel, but with a simpler kernel.

    Roberts: Uses diagonally oriented kernels for detecting diagonal edges.

    Canny: A multi-stage algorithm involving noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis.

Canny is considered the gold standard for edge detection due to its ability to detect weak and strong edges while suppressing noise. These techniques are foundational in object recognition, contour detection, image segmentation, and scene understanding.
🧪 Experiment 7: Sharpening using Gradient Mask
🔬 Theory (In-depth):

Sharpening enhances image features by emphasizing edges and transitions. It’s the opposite of smoothing.

    A gradient mask like the Sobel filter is used to detect directional changes in intensity (edges).

    By adding the result of the gradient to the original image, edges become more prominent.

This operation is useful in scenarios where image clarity is essential—like document analysis, microscopic imaging, and forensics. Over-sharpening, however, may amplify noise or introduce artifacts.
🧪 Experiment 9: DCT/IDCT Visualization
🔬 Theory (In-depth):

The Discrete Cosine Transform (DCT) decomposes an image into a sum of cosine functions at different frequencies.

    It is the cornerstone of JPEG compression: low-frequency components (containing most of the image information) are retained while high-frequency noise is discarded.

    The DCT basis functions are like templates—each representing a specific frequency and orientation.

    Visualizing these basis functions helps understand how DCT reconstructs images and why it's efficient for compression.

Understanding DCT is essential for work in compression algorithms, signal processing, watermarking, and Fourier analysis. This experiment provides a visual foundation for how DCT captures image energy in fewer coefficients
