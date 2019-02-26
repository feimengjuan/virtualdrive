ss1 = depth_read('2011_09_26_drive_0079_sync_image_0000000035_image_03_pred.png');
ss = depth_read('2011_09_26_drive_0079_sync_groundtruth_depth_0000000035_image_03.png');
sum = 0;
num = 0;
for i = 1:352
    for j = 1:1216
        if(ss(i,j)~=-1)
            sum = sum + (ss1(i,j)-ss(i,j))^2;
            num = num + 1;
        end
    end
end
sum = sum / num;
            