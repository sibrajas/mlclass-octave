function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

%tp=fp=fn=0;
predictions=(pval<epsilon);
#{
for i=1:size(pval,1),
tp=tp+((predictions(i)==1)&&(yval(i)==1));
fp=fp+((predictions(i)==1)&&(yval(i)==0));
fn=fn+((predictions(i)==0)&&(yval(i)==1));
end
#}
%tp=fp=fn=zeros(size(pval,1),1);
tp=sum((predictions==1)&(yval==1));
fp=sum((predictions==1)&(yval==0));
fn=sum((predictions==0)&(yval==1));
if(((tp+fp)==0)||((tp+fn)==0))
   prec=-1;
   recall=-1;
else
prec=tp/(tp+fp);
recall=tp/(tp+fn);
end
F1=(2*prec*recall)/(prec+recall);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
