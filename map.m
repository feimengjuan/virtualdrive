% ԭʼͼ����Ҫ���лҶȻ���������ֱ�Ӷ��Ҷ�ͼ
im = imread('2011_10_03_drive_0047_sync_image_0000000185_image_02.png');
img = rgb2gray(im);
im = double(img);
 
% ����ͼ��չ��Χ��0-255
gray = depth_read('2011_10_03_drive_0047_sync_image_0000000185_image_02_pred.png');
gray = uint8(gray);
 
% jet������ɫ����-����255-0����������jet(��ֵ)����210Ϊ��ɫ
cmap = colormap(jet(80));
rgb = ind2rgb(gray, cmap); 
 
map_image = uint8(im * 0.5 + rgb * 0.5 * 255);%�ȶ�ͼ��Ȩ�ش�Щ����ʾЧ������
imshow(map_image);
