function [himg, theta, radius] = circle_hough_trans(img)
% created by Vinay
% himg is the Accumulation matrix
% First, figure the max value for r (in positive direction).  This would be to the corner of the
% image (from the origin.  Solve using Pythagorean theorem
num_pos_r = ceil(sqrt(sum(size(img).^2)));

% we will store our values in himg...since r can be negative, we'll create
% an r-vector to keep track of what locations in himg correspond to the
% values of r and a theta vector that does the same

i_rng = size(img, 1);
j_rng = size(img, 2);

r = -1*num_pos_r:num_pos_r;
theta = 1:360;

radius = [48,50,52];
himg=zeros(i_rng, j_rng, length(radius));

for i=1:i_rng
  for j=1:j_rng
    if img(i,j) > 0 % Can be img(i,j)==1
        count = 0;
        for rad = radius
          count = count + 1;
          for ang=theta
            theta_loop=ang*pi/180; % change to radians  
            xloop = round(i+(rad*cos(theta_loop)));
            yloop = round(j+(rad*sin(theta_loop)));
            if xloop>0 && yloop>0 && xloop<=i_rng && yloop<=j_rng
                himg(xloop,yloop,count)=himg(xloop,yloop,count)+1;
            end
          end
        end
    end
  end
end
