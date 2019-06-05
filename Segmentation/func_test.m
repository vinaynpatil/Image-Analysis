% Getting all the file details from the distributed folder
files = dir('distributed/img*.ppm');
num_files = length(files);
%Getting a referance image to get the image size details for furture use
ref_img = double(imread(strcat('distributed/', files(1).name)));

gray_imgs = zeros([size(ref_img(:,:,1)),num_files]);
figure
colormap('gray');
for i=1:num_files
    img_loop = double(imread(strcat('distributed/', files(i).name)));
    img = 0.1*img_loop(:,:,1) + 0.85*img_loop(:,:,2) + 0.05*img_loop(:,:,3);
    %Applying median filter to removing noise from the data and preserving
    %edges.
    img = medfilt2(img,[3,3]);
    gray_imgs(:,:,i) = img;
    subplot(6,6,i)
    imagesc(img)
end

diskLocations = getOpticalDisk(gray_imgs);



function [circle_img] = getOpticalDisk(imgs)
img_bin = zeros(size(imgs));
for i=1:size(imgs,3)
    img_loop = imgs(:,:,i);
    
    diskLocations = imsharpen(img_loop,'Radius',2,'Amount',1);
    ksx = reshape([-1 -2 -1 0 0 0 1 2 1], 3, 3 );
    ksy = ksx';
    
    gx = conv2(diskLocations, ksx, 'same');
    gy = conv2(diskLocations, ksy, 'same');
    
    len_grad = sqrt(gx.^2 + gy.^2);
    edge_sol = len_grad>50;
    
    img_wind = bwmorph(edge_sol, 'dilate', 1);
    
    img_bin(:,:,i) = img_wind;
end

xrng = 1:size(img_loop, 2);
yrng = 1:size(img_loop, 1);
[xx,yy] = meshgrid(xrng,yrng);
figure
colormap('gray');
num_radii = zeros(1, size(imgs,3));
circle_img = zeros(size(imgs));
for i=1:size(imgs,3)
    img_loop = img_bin(:,:,i);
    [center, radii] = imfindcircles(img_loop==1, [50 120],'sensitivity', .98);
    num_radii(i) = radii(1);
    circle_img(:,:,i) = hypot(xx-center(1,1), yy-center(1,2))<=radii(1);
    subplot(6,6,i)
    imagesc(circle_img(:,:,i))
end

end
