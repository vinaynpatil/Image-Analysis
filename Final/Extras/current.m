% Final
% Name - Vinay Patil
% NetId - vpatil3

clear
clc

%%
%Setting the path for distributed data and test data
path_to_distributed_data = 'distributed';
path_to_test_data = 'distributed';

%Setting the labels for distributed data and test data
label_for_distributed_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]';
label_for_test_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]';

%Reading the image files from the distributed directory into an array
%called gray_imgs. Array size will be 605X700X36
files =  dir(sprintf('%s/img*.ppm',path_to_distributed_data));
num_files = length(files);
ref_img = im2double(imread(sprintf('%s/%s',path_to_distributed_data, files(1).name)));
gray_imgs = zeros([size(ref_img(:,:,1)),num_files]);

%Justification for selecting the below weights for the grayscale convertion

%Looking at the 3 channels of each images I realized that the blue channel
%contributes nothing(Except a dot occassionaly in the unhealthy retina)that
%will help us with the feature extraction.
%Similarly the red channel showed very low contrast when it comes to the
%features, but slightly better than the blue channel.
%The green channel turns out to be the best one among all as it provides
%the right contrast which makes our task of recognizing the features easy.
disp("Reading the image files from distributed data.....");
for i=1:num_files
    img_loop = im2double(imread(sprintf('%s/%s',path_to_distributed_data, files(i).name)));
    img = 0.1*img_loop(:,:,1) + 0.85*img_loop(:,:,2) + 0.05*img_loop(:,:,3);
    %Applying a 3X3 median filter to removing noise from the data and preserving
    %edges.Tried higher dimension median filter aswell, but it provided no benefits
    img = medfilt2(img,[3,3]);
    gray_imgs(:,:,i) = img;
end

%%
%The function definitions can be found at the end of this script

%Extracting my first feature (Exudates) from the provided distributed images.
feature1 = extractExudates(gray_imgs,num_files);

%Extracting my second feature (Hemorrhage) from the provided distributed images.
feature2 = extractHemorrhage(gray_imgs,num_files);

%Data preparation for cross validation using 2 classifiers(Logistic and KNN)
features = [feature1;feature2]';

%%

% plot(features(label_for_distributed_data==0,1), features(label_for_distributed_data==0,2), 'r.', 'markersize', 20)
% hold on
% plot(features(label_for_distributed_data==1,1), features(label_for_distributed_data==1,2), 'g.', 'markersize', 20)
% hold off

%The aim here is to balance the classes within each fold. I basically used
%the same code we used in class and tweeked a bit for the 6 fold CV
g1= repmat([1:6], 1,3);
g1 = g1(randperm(18));
g2 = repmat([1:6], 1,3);
g2 = g2(randperm(18));

cv_groups = [g1, g2];

pred = zeros(size(label_for_distributed_data)); % vector to hold predictions

%%

%CV with Logistic Classifier

disp("Running CV on distributed data using Logistic classifier.....");
%Looping through 6 folds i.e.considering 1 fold as a test set at a time and
%training on others
for fold=1:6
    %seperating into test and train
    test = features(cv_groups == fold,:);
    train = features(cv_groups ~= fold,:);
    
    labels_train = label_for_distributed_data(cv_groups ~= fold);
    
    % Train the classifier (logistic model fit)
    beta = glmfit(train, labels_train, 'binomial', 'link', 'logit');
    
    % Need to use the inverse logic to get the probabilities for test
    xb = [ones(size(test,1), 1), test]*beta;
    prob_test = exp(xb)./(1+exp(xb));
    pred_test = 1*prob_test>.5;
    
    pred(cv_groups == fold) = pred_test;
end

match = label_for_distributed_data == pred;

%Calculating the accuracies obtained
accuracy_class1 = mean(match(label_for_distributed_data == 0));
accuracy_class2 = mean(match(label_for_distributed_data == 1));
disp("-------------------------------------------------------------");
disp("CV Accuracies using Logistic Classifier");
disp("Healthy - "+num2str(accuracy_class1));
disp("Unhealthy - "+num2str(accuracy_class2));


%%

%CV with KNN Classifier

k=11;
pred = zeros(size(label_for_distributed_data)); % vector to hold predictions
disp("Running CV on distributed data using KNN classifier.....");
for fold=1:6
    %seperating into test and train
    test = features(cv_groups == fold,:);
    train = features(cv_groups ~= fold,:);
    
    labels_train = label_for_distributed_data(cv_groups ~= fold);
    
    nfeat = size(train, 2);
    
    for n=1:nfeat
        mn_train = mean(train(:,n));
        sd_train = std(train(:,n));
        train(:,n) = (train(:,n)-mn_train)/sd_train;
        test(:,n) = (test(:,n)-mn_train)/sd_train;
    end
    
    ntest = size(test, 1);
    ntrain = size(train, 1);
    pred_test = zeros(1, ntest);
    for i=1:ntest
        dist_from_train = sqrt(sum((ones(ntrain,1)*test(i,:)-train).^2, 2));
        [reord, ord] = sort(dist_from_train);
        knn = labels_train(ord(1:k));
        p_g1 = mean(knn == 0);
        p_g2 = mean(knn == 1);
        if (p_g2<p_g1)
            pred_test(i)=0;
        elseif (p_g1<p_g2)
            pred_test(i)=1;
        else
            %Doing a coin flip - Gives 0 or 1 randomly
            pred_test(i) = randperm(2,1) - 1;
        end
    end
    pred(cv_groups == fold) = pred_test;
end

%Calculating the accuracies obtained
match = label_for_distributed_data == pred;
accuracy_class1 = mean(match(label_for_distributed_data == 0));
accuracy_class2 = mean(match(label_for_distributed_data == 1));
disp("-------------------------------------------------------------");
disp("CV Accuracies using KNN Classifier");
disp("Healthy - "+num2str(accuracy_class1));
disp("Unhealthy - "+num2str(accuracy_class2));

%%

%Test data classification using Logistic Classifier

disp("-------------------------------------------------------------");
%Reading the image files from the test directory into an array
%called gray_imgs. Array size will be 605X700X36
files =  dir(sprintf('%s/img*.ppm',path_to_test_data));
num_files = length(files);
ref_img = im2double(imread(sprintf('%s/%s',path_to_test_data, files(1).name)));
gray_imgs = zeros([size(ref_img(:,:,1)),num_files]);
disp("Reading the image files from test data.....");
for i=1:num_files
    img_loop = im2double(imread(sprintf('%s/%s',path_to_test_data, files(i).name)));
    img = 0.1*img_loop(:,:,1) + 0.85*img_loop(:,:,2) + 0.05*img_loop(:,:,3);
    %Applying a 3X3 median filter to removing noise from the data and preserving
    %edges.
    img = medfilt2(img,[3,3]);
    gray_imgs(:,:,i) = img;
end

%%
%Using the same functions as before to extract the feature, the function
%definitions can be found at the end of this script
disp("Feature extraction for Test data.....");
test_feature1 = extractExudates(gray_imgs,num_files);
test_feature2 = extractHemorrhage(gray_imgs,num_files);

test_features = [test_feature1;test_feature2]';

%%
%Code to run classification on the test data follows below
%I am making use of logistic classifier for this task

disp("Running classifier on the test data using Logistic classifier.....");
test = test_features;
train = features;

labels_train = label_for_distributed_data;

% Train the classifier (logistic model fit)
beta = glmfit(train, labels_train, 'binomial', 'link', 'logit');

% Need to use the inverse logic to get the probabilities for test
xb = [ones(size(test,1), 1), test]*beta;
pred = exp(xb)./(1+exp(xb));
pred = 1*pred>.5;

match = label_for_test_data == pred;

accuracy_class1 = mean(match(label_for_test_data == 0));
accuracy_class2 = mean(match(label_for_test_data == 1));

disp("Classification Accuracies using Logistic Classifier");
disp("Healthy - "+num2str(accuracy_class1));
disp("Unhealthy - "+num2str(accuracy_class2));


%%
%--------------------------------------------------------------------------
%Function definitions start here

%Function -  To extract exudates
%Input - The read gray images and the count of images(num_files)
%Output - Feature extracted for each image as a vector
function feature =  extractExudates(gray_imgs,num_files)
%feature variable will hold the feature value extracted from each of the image supplied
feature = zeros(1,num_files);
disp("Extracting exudates - feature 1.....");
%Please uncomment the below code to generate a 6x6 panel plot
%figure
%colormap('gray');
for i=1:36
    %The illumination gradient was really hurting me on the healthy images,
    %so I had a figure out a way to overcome this problem. Went through the
    %a lot of papers to find a solution to this problem and every
    %literature seem to suggest that, enhancing the contrast of the image
    %should fix the problem. This is when I came across the adapthisteq
    %function. This function comes built in with MATLAB as part of Image
    %Processing Toolbox.
    %Using this function helped me bring out the exudates in the unhealthy
    %images.
    %From what I gather this particular function enhances the contrast of
    %the images by transforming the image intensity, which is done by operating
    %on a small regions instead on the whole image. Such small regions
    %are then combined using bilinear interpolation in order to eliminate
    %artificially induced boundaries
    %Along with the image, it can take various other key value pair
    %parameters.
    %The parameter ClipLimit can we made use to limit the contrast, it
    %ranegs between 0 to 1. I tried various values for different images at
    %hand and found the default value of 0.01 to work out best.
    %We can also set the no of bins for the histogram using NBins, using
    %which we can control the dynamic range of values.
    %Also this function changed my image pixel range to 0-1
    img = adapthisteq(gray_imgs(:,:,i));
    
    %Initially I thought of removing the optical disk from all the retina
    %images(Since they share similar level of intensities and color), but
    %later realized that since all the retina images have the optical disk, it
    %will just result in increasing the exudates count by 1 for all the images!
    %Hence this can be safely ignored.
    
    %Looking at the grayscaled images of the retina, it seems that windowing
    %method would be best suitable here to get the parts of the image I am most
    %interested in
    %I noticed that since all images may not have a similar intensity
    %distribution, I could not set a common value for min and max, instead
    %I used the range of magnitudes in that image to find a threshold.
    %The below set values seem to work best across the stack of images.
    %Even though a few of the unhealthy retina images are missing some
    %exudates, I didn't want to overfit the data. I followed your
    %advice to select 4/6 images and use just those while working on the
    %feature, this seem to give me decent result
    min_val = 0.9*max(img(:));
    max_val = max(img(:));
    slope = 1/(max_val - min_val);
    int = -1*min_val*1/(max_val - min_val);
    rng = img>min_val & img<max_val;
    img_wind = 0*img;
    img_wind(rng) = img(rng)*slope + int;
    
    %Once I applied windowing to the retina images, I noticed that there
    %were some loosely connected components around the optic disk, this if
    %left untreated would introduce unnecessary additions to the exudates
    %counts for that image. And hence I applied erode to the windowed
    %image, which worked quite well in removing those loosely connected
    %component, I also noticed that, this result improved with the increased
    %application of erode, but it came with a minor downside, as we know
    %that erode makes the object thinner, the recognized exudates which were very small,
    %seem to disappear, hence I had to settle with 2 applications of erode.
    img_wind = bwmorph(img_wind, 'erode', 2);
    
    %Now the final step is to count the no of exudates extracted in the
    %retina image, but before doing so, I wanted to make the bright regions
    %i.e. exudates brighter, which would later help to count them easily.
    %Hence I used the morphological close operation(imclose) on the image, 
    %which is nothing but applying dilation followed by an erosion operation.
    %imclose function also takes in the structuring element as an argument.
    %Here I provided the disk as a structuring element with a radius of 15.
    %As we typically choose a structuring element the same size and shape as 
    %the objects we want to process in the input image, I made use of the
    %imtool function to measure the approximate radius of the exudates(using
    %the scale it provides) and 15 seems to fit most of them.
    SE = strel('disk',15);
    img_wind = imclose(img_wind,SE);
    
    %Now since the contrast is fairly good(binary), we can make use of the
    %bwlabel function to label the connected components and count the no of
    %objects labelled, which is one of the outputs retrned by the function.
    %To determine the no of connected components in the image,I also noticed
    %that the four functions i.e. bwlabel,BWLABELN,BWCONNCOMP and REGIONPROPS
    %all are useful in computing the connected components for binary images.
    %I used bwlabel as I was familiar with the fuction. But I made a note to
    %make use of BWCONNCOMP in the future as it the most recent addition, faster 
    %and is intended to replace the other functions I mentioned. For bwlabel, the
    %input must be a 2D matrix, and the values can be either logical or
    %numeric, but using either doesn't seem to make a difference to the
    %speed of the algorithm.
    %The bwlabel function makes use of the region growing algorithm we learnt in
    %class. It picks a random strting point and for each of the starting
    %point it picks, it takes a look into the neighborhood, which is set to
    %8-connected objects by default, but provides an option to change it to
    %4-connected objects using the section argument to the function (N). Each 
    %pixel from the neighborhood is added to the mask if it meets the criteria.
    %The background is labeled as 0 and the recognized objects are labeled
    %in increments of 1.
    %This particular convention helps in visualizing the coloring for each
    %object(to distinguish them) with the help of the function label2rgb.
    [~, numberOfObject] = bwlabel(img_wind);
    
    %Saving the count of the exudates in the feature vector.
    feature(i) = numberOfObject;
    
    %Please uncomment the below code to generate a 6x6 panel plot
    %subplot(6,6,i);
    %imagesc(img_wind);
end
disp("Exudates extracted!.....");
end


%Function - To extract hemorrhages
%Input - The read gray images and the count of images(num_files)
%Output - Feature extracted for each image as a vector
function feature =  extractHemorrhage(gray_imgs,num_files)

feature = zeros(1,num_files);
disp("Extracting hemorrhages - feature 2.....");
%Please uncomment the below code to generate a 6x6 panel plot
%figure
%colormap('gray');
for i=1:36
    %The first thing I noticed in the grayscale images, when I was looking
    %for hemorrhages was that they are dark in color and hence were not
    %easy to extract by some thresholding. So the idea  am using here is to
    %try to change the dark pixels to the bright pixels and vice-versa. So 
    %the best function that helps accomplish this task is the imcomplement
    %function. The idea behind this function is simple, if the image fed to
    %this function is binary, then white becomes black and black becomes
    %white. ANd if a gray image is fed to it instead, then the dark areas
    %become lighter  and light areas become brighter.
    img = imcomplement(gray_imgs(:,:,i));
    
    %After the complement process I noticed that there were intensity
    %variations in the background of the retina images and relaized this
    %might degrade the process of feature extraction. Going through the
    %literature I came across this paper - (Automatic Microaneurysm Detection
    %and Characterization Through Digital Color Fundus Images, 
    %https://www.osti.gov/servlets/purl/979799), where they mention that
    %the shade correction of the image should mitigate this issue. The idea
    %was to take the background of the gray image, by smoothing the gray image 
    %with the application of median filter, here the suggested window size
    %was 25X25. I tried playing out with this and found that smaller the window
    %size, the background would contain important information from the
    %image, on the contrary if a higher window size was used, the
    %processing time increased with no performance improvements. Once I got
    %the background for the retina image under consideration, I subtarted
    %it from the gray image to achieve shade correction.
    bg = medfilt2(img,[25,25]);
    img = img - bg;
    
    %The next step of the process was to fix the contrast of the image, 
    %as in the current image hemorrhages were still not the brightest,
    %now this was accomplished using the function adapthisteq, working of
    %which was explained in the function - extractExudates.
    img = adapthisteq(img);
    
    %Now looking at the result I noticed that the region of
    %interest is fairly visible, but I didn't want my bwlabel function
    %picking up the dark regions(Exudates and optical disk) as the
    %potential objects of interest, hence my next move ws to completely
    %eliminate them from the picture, which was accomplished by binarizing
    %the image. Now to do this, I first thought of manually setting the
    %threshold for the pixel values which are to be set to 1, but it didn't
    %quite work nicely for all the images(Threshold I tried was 0.3-0.6).
    %So instead I made use of the imbinarize function to accomplish the
    %task.The function imbinarize binarizes the grayscale images using a
    %global threshold computed using the Otsu's method which separates forground from
    %background. This method works by trying all possible threshold and
    %calculating the variance of image intensities for the values above and
    %below that threshold. The cost for that split is then calculated as
    %pecent{below}x var{below} + pecent{above}*var{above}, where percent{below} is the
    %percentage of the values below threshold and var{below} is the variance of
    %the image intensities below that threshold.  The split with the lowest
    %cost wins. The function imbinarize along with the global method
    %provides and option to make use of the adaptive method, which acts by
    %choosing the local first order image statistics around each pixel.
    %Also by default the imbinarize function assumes that the foreground is
    %brighter than the background, which is what I wanted it to do.
    img = imbinarize(img);
    
    %Now you might have noticed that when I first took a complement of the
    %grayscale images, it not only brightned the hemorrhage, but optical
    %vessels/nerves as well. Now the first thing that came to my mind was
    %to separate out the nerves from the image and then subtract them from
    %the grayimage, so that just the hemorrhage are left out in the image.
    %To do this, I went back to the literature - https://www.osti.gov/servlets/purl/979799
    %which suggested me to run several morphological operations(Specifically thrice
    %with a alternating opening and closing, with linear-structure 15,15 and 29), 
    %using a set of linear structuring elements, so that this operation
    %removes every isolated round and bright zones whose diameter was less
    %than some specified size(I took 15 for the similar reasons I mentioned 
    %in exudates extraction). This method does segment out the optic
    %nerves/vessels but I noticed that because of the morphological
    %operations performed on the images, the places where the nerves had a
    %circular curvature were removed as well which ended up creates new
    %circular disks on my final image when I subtracted the optical nerves
    %from my original image(which had hemorrhage and optic nerves/vessels).
    %Since that method was not fruitful, I ran the dilate method on my
    %image with an aim to recognize the optic nerves as a single component.
    %I was able to get a fairly good result on the sample images I was 
    %working with(to avoid overfitting) by running it 4 times on the
    %image. Surprisingly most of the hemorrhages were fairly far away from
    %each other so as not get pulled into other hemorrhages(Even thought some
    %did). I know this might not be what you expected, but I spent a 
    %lot of time extracting the hemorrhage trying different method in papers,
    %but almost all of them suggested the method of removing the nerves beforehand,
    %which for some reason never worked for me!   
    img = bwmorph(img, 'dilate', 4);
    
    %For the final step the option was to either compute the area covered
    %by the feature or to count the no of objects in the image. Computing
    %the area covered didn't make sense to me because of the optical nerves
    %in the images. 
    %Hence I resorted to counting them. Initially I tried making use of the
    %circular hough function to count the no of circular object in the images,
    %but the problem here was the curvature in the nerves were considered
    %as the hemorrhages, which gave me bad results as a lot of healthy
    %retina images have a curvatures in their nerves.
    %Hence I resorted to counted them using the bwlabel function I
    %described before. This seem to work ok by counting the
    %dilated nerve as a single connected component in most of the images.
    [~, numberOfObject] = bwlabel(img);
    
    %Saving the count of the hemorrhages in the feature vector.
    feature(i) = numberOfObject;
    
    %Please uncomment the below code to generate a 6x6 panel plot
    %subplot(6,6,i);
    %imagesc(img);
end
disp("Hemorrhages extracted!.....");
end