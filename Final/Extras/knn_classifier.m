feat_lecture = [13     1     1     1     1     1     1     1     1     1     1     3     1     3     3     1     1  1     3     2     5     6     1     1     1     1     2     3     9     1     4     1     4     8   4     3;35    33    40    39    30    30    59     4    27    45    64    38    45    52    44    27    32  44    59    94    50     4    63    62    40    54    25    48   102    56    67    57    74     4  55    67]';

labels_lecture = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]';

plot(feat_lecture(labels_lecture==0,1), feat_lecture(labels_lecture==0,2), 'r.', 'markersize', 20)
hold on
plot(feat_lecture(labels_lecture==1,1), feat_lecture(labels_lecture==1,2), 'g.', 'markersize', 20)
hold off

feat1 = feat_lecture(:,1);
feat2 = feat_lecture(:,2);


%%
% 2-fold CV Logistic regression

% This will look familiar, since I'm using the same general structure as I
% used for KNN & lDA

% First look at labels:  We want our data to be balanced within each fold
labels_lecture
% Logistic regression uses 1/0 as dependent variable, so we will change
%labels_lecture = labels_lecture -1;


% I'm being lazy here.  You should randomly select the data for your folds.
%  You will adapt the homework to do so.
%cv_groups = [ones(1, 25), 2*ones(1, 25), ones(1, 25), 2*ones(1, 25)];

g1= repmat([1:6], 1,3);
g1 = g1(randperm(18));
g2 = repmat([1:6], 1,3);
g2 = g2(randperm(18));
    
cv_groups = [g1, g2];

k=10;
pred = zeros(size(labels_lecture)); % vector to hold predictions

for fold=1:6
    %seperating into test and train
  test = feat_lecture(cv_groups == fold,:);  
  train = feat_lecture(cv_groups ~= fold,:);
 
  labels_train = labels_lecture(cv_groups ~= fold);
  
  nfeat = size(train, 2);
        % Normalization: Can do this without a loop, but I wanted it to be clear
        for n=1:nfeat
            mn_train = mean(train(:,n));
            sd_train = std(train(:,n));
            train(:,n) = (train(:,n)-mn_train)/sd_train;
            
            test(:,n) = (test(:,n)-mn_train)/sd_train;
        end
        
  %Similar to previous code
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
          % Fix this to either be a coin flip 
          pred_test(i)=randperm(2,1) - 1; 
      end   
  end
  pred(cv_groups == fold) = pred_test;
end

%Comparing true with predicted labels
match = labels_lecture == pred;
accuracy_g1 = mean(match(labels_lecture == 0))
accuracy_g2 = mean(match(labels_lecture == 1))

%  Now adapt the above code for K folds and randomly sample the data, while
%  keeping labels balanced for each fold

