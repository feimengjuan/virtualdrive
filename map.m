% 原始图像，需要进行灰度化，我这里直接读灰度图
im = imread('2011_09_26_drive_0079_sync_image_0000000035_image_03.png');
img = rgb2gray(im);
im = double(img);
 
% 概率图扩展范围是0-255
gray = depth_read('2011_09_26_drive_0079_sync_image_0000000035_image_03_pred.png');
gray = uint8(gray);
 
% jet定义颜色，红-蓝：255-0，可以设置jet(阈值)调整210为红色
cmap = colormap(jet(80));
rgb = ind2rgb(gray, cmap); 
 
map_image = uint8(im * 0.5 + rgb * 0.5 * 255);%热度图的权重大些，显示效果更好
imshow(map_image);
