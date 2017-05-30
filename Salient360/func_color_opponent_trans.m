function img_oppo = func_color_opponent_trans(img_in)

% transform the RGB space to opponent space

% split color channels
R = double( img_in(:,:,1) );
G = double( img_in(:,:,2) );
B = double( img_in(:,:,3) );

% derivatives in opponent color space
o1 = (R - G) / sqrt(2);
o2 = (R + G - 2*B) / sqrt(6);
o3 = (R + G + B) / sqrt(3);

% normalize
img_oppo( :,:,1 ) = uint8( ( o1 - min(o1(:)) ) / ( max(o1(:)) - min(o1(:)) ) * 255 );
img_oppo( :,:,2 ) = uint8( ( o2 - min(o2(:)) ) / ( max(o2(:)) - min(o2(:)) ) * 255 );
img_oppo( :,:,3 ) = uint8( ( o3 - min(o3(:)) ) / ( max(o3(:)) - min(o3(:)) ) * 255 );