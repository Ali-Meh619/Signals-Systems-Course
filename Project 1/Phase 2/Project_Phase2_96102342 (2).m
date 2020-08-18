%% Filter Design
clc;
load('SubjectData1.mat');
subplot(1,2,1);
sample_data = train(2,:);
no1 = floor(length(train(3,:))/16);
h=impz(filter_1);
plot(sample_data(1:no1),'k');
hold on
filtered_1 = zphasefilter(h,train(3,:));
no2 = floor(length(zphasefilter(h,train(3,:)))/12);
plot(filtered_1(1:no2),'r');
subplot(1,2,2);
no3 = floor(length(train(3,:))/10);
l=train(3,:);
plot(sample_data(1:no3),'k');
h=impz(filter_1);
no4 = floor(length(conv(train(3,:),h))/10);
filtered_2 = conv(sample_data,h);
hold on
plot(filtered_2(1:no4),'r');

% comparing groupdelay with grpdelay
figure
plot(groupdelay(h',8000))
figure
plot(grpdelay(h',8000))
%% DataSet and initial processes

for i=1:9
w=subject.Test(i).test;

    for j=1:size(w,2)
       
        if w(10,j)==13
            
            type(1,i)="SC";
            
        
    end
    
    
    end
end
%%
type(1,1)="SC";
type(1,2)="SC";
type(1,3)="RC";
type(1,4)="RC";
type(1,5)="RC";
type(1,6)="RC";
type(1,7)="RC";
type(1,8)="RC";
type(1,9)="RC";

%% EstekhraJe ViJegI

target_counter = 0;
nontarget_counter = 0;
X = [];
Y = [];
    
trial_numbers = size(subject.Time(1).Time.Target,2) + size(subject.Time(1).Time.Non_Target,2);
tt=zeros(2,trial_numbers);
max_target = size(subject.Time(1).Time.Target,2);
max_nontarget = size(subject.Time(1).Time.Non_Target,2);

ta=zeros(1,max_target);
non=zeros(1,max_nontarget);
trial_times = [];
for i = 1 : trial_numbers
    if  (nontarget_counter == max_nontarget) || ((target_counter < max_target) && (subject.Time(1).Time.Target(target_counter+1) < subject.Time(1).Time.Non_Target(nontarget_counter+1)))
        target_counter = target_counter + 1;
        Y = [Y;1];
        dd(1,i)=1;
        ta(1,target_counter)=target_counter;
        current_epoch = f2(1).Taeget(:,:,target_counter);
        trial_times = [trial_times;(subject.Time(1).Time.Target(target_counter))];
    else
        nontarget_counter = nontarget_counter + 1;
        Y = [Y;0];
        dd(2,i)=1;
        non(1,nontarget_counter)=nontarget_counter;
        current_epoch = f2(1).Non_Taeget(:,:,nontarget_counter);
        trial_times = [trial_times;(subject.Time(1).Time.Non_Target(nontarget_counter))];
    end
    
    
    ce=f2(1).Test(:,:,i);
    feature_numbers = 0;
    fff=0;
    q=1;
    for j = 2 : 9  %calculating Mean 1 ta 9 
       X(i,j) = mean(current_epoch(j,:));
       XX(i,j) = mean(ce(j,:));
       feature_numbers = feature_numbers + 1;
       fff=fff+1;
    end
    feature.mean=q:q+fff;
    q=q+fff+1;
    for j = 2 : 9 %calculating Mode 10 to 18
        new_feature = mode(current_epoch(j,:));
        nf = mode(ce(j,:));
        feature_numbers = feature_numbers + 1;
        fff=fff+1;
        X(i,feature_numbers) = new_feature;
        XX(i,fff)=nf;
    end
    feature.mode=q:q+fff;
    q=q+1+fff;
     for j = 2 : 9 %calculating VAR 19 to 27
        new_feature = var(current_epoch(j,:));
        nf = var(ce(j,:));
        feature_numbers = feature_numbers + 1;
        fff=fff+1;
        X(i,feature_numbers) = new_feature;
         XX(i,fff)=nf;
     end
    feature.var=q:q+fff;
    q=q+1+fff;
    for j = 2 : 9  %time_domain signal 28 to 1953
       new_feature = current_epoch(j,:);
        
        nf = ce(j,:);
        ff=size(nf,2);
       n = size(new_feature,2);
       
       feature_numbers = feature_numbers + n;
       fff=fff+ff;
       QQ(i,1:size(new_feature,2)) = new_feature;
       X(i,feature_numbers-n+1:feature_numbers) = new_feature;
        XX(i,fff-ff+1:fff)=nf;
    end
    feature.timedomain=q:q+fff;
    q=q+1+fff;
    for j = 2 : 8  %correlation between channels 1954 to 1989
        for k = j:9
            new_feature = corr(current_epoch(j,:)',current_epoch(k,:)');
            nf = corr(ce(j,:)',ce(k,:)');
            fff=fff+1;
            feature_numbers = feature_numbers + 1;
            X(i,feature_numbers) = new_feature;
            
            XX(i,fff) = nf;
            
        end
    end
    
    feature.corr=q:q+fff;
    q=q+1+fff;
    for j = 2 : 9  % mean fequancy 1990 to 1998
       new_feature = meanfreq(current_epoch(j,:),250);
       nf = meanfreq(ce(j,:),250);
       fff=fff+1;
       feature_numbers = feature_numbers + 1;
       X(i,feature_numbers) = new_feature;
       XX(i,fff) = nf;
    end
    feature.mean_freq=q:q+fff;
    q=q+1+fff;
    
    for j = 2 : 9  % mode fequancy 1999 to 2007
       new_feature = medfreq(current_epoch(j,:),250);
       nf = medfreq(ce(j,:),250);
       fff=fff+1;
       feature_numbers = feature_numbers + 1;
       X(i,feature_numbers) = new_feature;
       XX(i,fff) = nf;
       
    end
    feature.med_freq=q:q+fff;
    q=q+1+fff;
    
    for j = 2 : 9  %Discrete sine Transform signal 2008 to 3933
       new_feature = dst(current_epoch(j,:));
       nf = dst(ce(j,:));
       n = size(new_feature,2);
       ff=size(nf,2);
       fff=fff+ff;
       feature_numbers = feature_numbers + n;
       X(i,feature_numbers-n+1:feature_numbers) = new_feature;
        XX(i,fff-ff+1:fff) = nf;
    end
    
    feature.dst=q:q+fff;
    q=q+1+fff;
    
    for j = 2 : 9  %Discrete cosine Transform signal 3934 to 5859
       new_feature = dct(current_epoch(j,:));
       nf = dct(ce(j,:));
       n = size(new_feature,2);
       ff=size(nf,2);
       fff=fff+ff;
       feature_numbers = feature_numbers + n;
       X(i,feature_numbers-n+1:feature_numbers) = new_feature;
        XX(i,fff-ff+1:fff) = nf;
       
    end
    feature.dct=q:q+fff;
    q=q+1+fff;
    
end

%%
pt=1;
pn=1;

ss=feature_numbers;
bb=fff;
for i=1:trial_numbers

    if dd(1,i)==1
        current_epoch = f2(1).Taeget(:,:,ta(1,pt));
        pt=pt+1;
    else
        current_epoch = f2(1).Non_Taeget(:,:,non(1,pn));
        pn=pn+1;
    end
     ce=f2(1).Test(:,:,i);
    feature_numbers=ss;
    fff=bb;
    for j= 2 : 9  %wavelet transform
     [new_feature1,new_feature2]=dwt(current_epoch(j,:),'haar');
     [nf1,nf2]=dwt(ce(j,:),'haar');
     n=size(new_feature1,2)+size(new_feature2,2);
     ff=size(nf1,2)+size(nf2,2);
     feature_numbers = feature_numbers + n;
     fff=ff+fff;
     X(i,feature_numbers-n+1:feature_numbers)=[new_feature1,new_feature2];
     XX(i,fff-ff+1:fff)=[nf1,nf2];
    end
    
end
feature.wavelet=q:q+fff;
    q=q+1+fff;
%%
pt=1;
pn=1;
ss=feature_numbers;
bb=fff;
for i=1:trial_numbers

    if dd(1,i)==1
        current_epoch = f2(1).Taeget(:,:,ta(1,pt));
        pt=pt+1;
    else
        current_epoch = f2(1).Non_Taeget(:,:,non(1,pn));
        pn=pn+1;
    end
    
    feature_numbers=ss;
    fff=bb;
    ce=f2(1).Test(:,:,i);
      for j= 2 : 9 %Energy Freq_Band
    new_feature=Freqband(current_epoch(j,:),4,7,256);%theta
    nf=Freqband(ce(j,:),4,7,256);
    fff=fff+1;
    feature_numbers = feature_numbers + 1;
     X(i,feature_numbers) = new_feature;
     XX(i,fff) = nf;
     feature.E_theta=q;
    q=q+1;
     
      new_feature=Freqband(current_epoch(j,:),8,12,256);%alpha
      nf=Freqband(ce(j,:),8,12,256);
      fff=fff+1;
    feature_numbers = feature_numbers + 1;
     X(i,feature_numbers) = new_feature;
      XX(i,fff) = nf;
      feature.E_alpha=q;
     q=q+1;
      
      new_feature=Freqband(current_epoch(j,:),14,37,256);%beta
       nf=Freqband(ce(j,:),14,37,256);
    feature_numbers = feature_numbers + 1;
    fff=fff+1;
     X(i,feature_numbers) = new_feature;
         XX(i,fff) = nf;
         feature.E_beta=q;
    q=q+1;
         
    end
end


%%
pt=1;
pn=1;

ss=feature_numbers;
bb=fff;
for i=1:trial_numbers

    if dd(1,i)==1
        current_epoch = f2(1).Taeget(:,:,ta(1,pt));
        pt=pt+1;
    else
        current_epoch = f2(1).Non_Taeget(:,:,non(1,pn));
        pn=pn+1;
    end
    ce=f2(1).Test(:,:,i);
    feature_numbers=ss;
    fff=bb;
    for j= 2: 9 %STFT
    new_feature=spectrogram(current_epoch(j,:)); 
    nf=spectrogram(ce(j,:)); 
    n=size(new_feature,1)*size(new_feature,2);
    ff=size(nf,1)*size(nf,2);
    fff=fff+ff;
    feature_numbers = feature_numbers + n;
    X(i,feature_numbers-n+1:feature_numbers)=abs(transpose(new_feature(:))); 
    XX(i,fff-ff+1:fff)=abs(transpose(nf(:)));
    feature.stft_abs=q:q+fff;
    q=q+1+fff;
    
    fff=fff+ff;
    feature_numbers = feature_numbers + n;
    X(i,feature_numbers-n+1:feature_numbers)=angle(transpose(new_feature(:))); 
     XX(i,fff-ff+1:fff)=angle(transpose(nf(:)));
     feature.stft_angle=q:q+fff;
    q=q+1+fff;
    end
    
end
%%

pt=1;
pn=1;
ss=feature_numbers;
bb=fff;
for i=1:trial_numbers

    if dd(1,i)==1
        current_epoch = f2(1).Taeget(:,:,ta(1,pt));
        pt=pt+1;
    else
        current_epoch = f2(1).Non_Taeget(:,:,non(1,pn));
        pn=pn+1;
    end
    
    feature_numbers=ss;
    fff=bb;
        ce=f2(1).Test(:,:,i);
      for j= 2 : 9 %Wentropy
    new_feature=wentropy(current_epoch(j,:),'shannon');%shannon
    nf=wentropy(ce(j,:),'shannon');
    fff=fff+1;
    feature_numbers = feature_numbers + 1;
     X(i,feature_numbers) = new_feature;
      XX(i,fff) = nf;
      feature.wentropy_shannon=q;
    q=q+1;
      new_feature=wentropy(current_epoch(j,:),'log energy');%log energy
       nf=wentropy(ce(j,:),'log energy');
     feature_numbers = feature_numbers + 1;
     fff=fff+1;
     X(i,feature_numbers) = new_feature;
     XX(i,fff) = nf;
         feature.wentropy_logenergy=q:q+fff;
    q=q+1;     
    end
end
%% Coping target rows
clc;

num_new_rows = floor(max_nontarget * 2);
num_column = size(X,2);
X_copied = zeros(num_new_rows,num_column);
Y_copied = zeros(num_new_rows,1);
vec1 = randperm(num_new_rows,trial_numbers);
vec1_sorted = sort(vec1);
X_copied(vec1_sorted,:) = X;
Y_copied(vec1_sorted,:) = Y;
empty_rows = [];
Target_rows = X(find(Y),:);
for i = 1:num_new_rows
    
    for j=1:size(vec1,2)
    if i==vec1(1,j)
        empty_rows = [empty_rows i];
    end
    end
end


Y_copied(empty_rows) = ones(size(empty_rows,2),1);
eachtarg_cnt = 0;
eachtarg_num = floor(size(empty_rows,2)/max_target);
targ_pointer = 1;
for i = 1 : size(empty_rows,2)
    X_copied(empty_rows(i),:) = Target_rows(targ_pointer);
    eachtarg_cnt = eachtarg_cnt + 1;
    if eachtarg_cnt == eachtarg_num
        targ_pointer = targ_pointer + 1;
        eachtarg_cnt = 0;
    end
end

%% Calculating J_Matrix
J = [];

for i = 1 : size(X,2)
   target_values = [];
   nontarget_values = [];
   for j = 1 : floor((0.75)*size(X,1))
      if Y(j) == 1 
         target_values = [target_values X(j,i)];
      else
         nontarget_values = [nontarget_values X(j,i)];
      end
   end
   M = mean(X(:,i));
   M0 = mean(target_values);
   M1 = mean(nontarget_values);
   S0 = var(target_values);
   S1 = var(nontarget_values);
   j_value = (abs(M-M0)+abs(M-M1))/(S0+S1);
   J = [J j_value];
end
J_sorted = sort(J);

%% Cross_Validation
clc;
%XX=RR;
OPO=X;
UU=XX;

%%
clc;
max_precision = 0;
jvalue_threshold = 0;
Accuracy = [];
Accuracy_balanced = [];

for percent = 0.01:0.025:0.76 %retain that percent of features
   X_updated = X;
   Y_updated = Y;
   threshold = J_sorted(floor(size(J,2)-percent*size(J,2)));
   deletion_column = [];
   for i = 1 : size(X_updated,2) %deleting unwanted features
       if J(i) <= threshold
          deletion_column = [deletion_column i]; 
       end
   end
   X_updated(:,deletion_column) = [];
   X_training = X_updated(1:floor((0.75)*size(X_updated,1)),:);
   X_Validation = X_updated(floor((0.75)*size(X_updated,1))+1:size(X_updated,1),:);
   
   Y_training = Y_updated(1:floor((0.75)*size(Y_updated,1)));
   Y_Validation = Y_updated(floor((0.75)*size(Y_updated,1))+1:size(Y_updated,1));
   
   
   SVM_Model =fitcsvm(X_training,Y_training,'classnames',[1 0],'cost',[0 35;1 0]);
   label = predict(SVM_Model,X_Validation);
   
   n1 = 0;   % Number_of_target_inValidation
   n2 = 0;   % Number_of_nontarget_inValidation
   
   
   
   
   for i = 1 : size(Y_Validation,1)
      if Y_Validation(i) == 1
          n1 = n1 + 1;
      else
          n2 = n2 + 1;
      end
   end
   false_target = 0;
   false_nontarget = 0;
   True = 0;
   for i = 1 : size(Y_Validation,1)
       if Y_Validation(i) == label(i)
           True = True + 1; 
       elseif Y_Validation(i) == 1
           false_target = false_target + 1;
       else
           false_nontarget = false_nontarget + 1;
       end    
   end
   precision = (True/size(Y_Validation,1))*100;
   Accuracy = [Accuracy precision];
   fault_balanced = (false_target*35 + false_nontarget)/(n1*35 + n2);
   precision_balanced = (1-fault_balanced)*100;
   Accuracy_balanced = [Accuracy_balanced precision_balanced];
   
   if precision_balanced > max_precision
       max_precision = precision_balanced;
       percent_max = percent;
       threshold_optimized = threshold;
       X_optimized = X_updated;
       delation=[];
       delation=deletion_column;
   end
end



plot(Accuracy_balanced)


%% implementation Of Machine LearNing
% SVM predict
RR=XX;
%%
clc;
XX(:,delation)=[];

SVM_Model = fitcsvm(X_optimized,Y_updated,'classnames',[1 0],'cost',[0 35;1 0]);
X_test = XX;
predicted_Y_SVM = predict(SVM_Model,X_test);
charachters = zeros(2,36);  % Table of 36 charachter  SC
letter_counter = 0;
letters = [];
for i = 1 : size(trial_times,1)  % guess the word with SVM
   char_number = subject.Train.train1(10,trial_times(i));
   charachters(1,char_number) = charachters(1,char_number) + 1;
   if predicted_Y_SVM(i) == 1
       charachters(2,char_number) = charachters(2,char_number) + 1;
   end
   if sum(charachters(1,:)) == 15*36
       current_letter = 1;
       target_number_max = charachters(2,1);
       for j = 2 : 35
           if charachters(2,j) > target_number_max
               target_number_max = charachters(2,j);
               current_letter = j;
           end
       end
       letters = [letters current_letter];
       letter_counter = letter_counter + 1;
       charachters = zeros(2,36);
   end
end

%% RC paradigm , word detection     -----> SVM
XX(:,delation)=[];

SVM_Model = fitcsvm(X_optimized,Y_updated,'classnames',[1 0],'cost',[0 35;1 0]);
X_test = XX;
predicted_Y_SVM = predict(SVM_Model,X_test);
vectors = zeros(2,12);  % Table of 12 row & column  SC
letter_counter = 0;
letters = [];
for i = 1 : size(trial_times,1)  % guess the word with SVM
   RorC_number = subject.Train.train4(10,trial_times(i));
   vectors(1,RorC_number) = vectors(1,RorC_number) + 1;
   if predicted_Y_SVM(i) == 1
       vectors(2,RorC_number) = vectors(2,RorC_number) + 1;
   end
   if sum(vectors(1,:)) == 15*12
       current_column = 1;
       current_row = 7;
       target_number_max_row = vectors(2,1);
       target_number_max_column = vectors(2,7);
       for j = 2 : 6
           if vectors(2,j) > target_number_max_column
               target_number_max_column = vectors(2,j);
               current_column = j;
           end
       end
       for j = 8 : 12
           if vectors(2,j) > target_number_max_row
               target_number_max_row = vectors(2,j);
               current_row = j;
           end
       end
       letters = [letters;current_row current_column];
       letter_counter = letter_counter + 1;
       vectors = zeros(2,12);
   end
end
%% LDA predict
clc;

LDA_Model = fitcdiscr(X_optimized,Y_updated,'classnames',[1 0],'cost',[0 35;1 0]);
X_test = X_optimized;
predicted_Y_LDA = predict(LDA_Model,X_test);
charachters = zeros(2,36);  % Table of 36 charachter
letter_counter = 0;
letters = [];

for i = 1 : size(trial_times,1)  % Guess the word with LDA
   char_number = subject.Train.train1(10,trial_times(i));
   charachters(1,char_number) = charachters(1,char_number) + 1;
   if predicted_Y_LDA(i) == 1
       charachters(2,char_number) = charachters(2,char_number) + 1;
   end
   if sum(charachters(1,:)) == 15*36
       current_letter = 1;
       target_number_max = charachters(2,1);
       for j = 2 : 35
           if charachters(2,j) > target_number_max
               target_number_max = charachters(2,j);
               current_letter = j;
           end
       end
       letters = [letters current_letter];
       letter_counter = letter_counter + 1;
       charachters = zeros(2,36);
   end
end

%% RC paradigm , word detection  ------> LDA
vectors = zeros(2,12);  % Table of 12 row & column  SC
letter_counter = 0;
letters = [];
for i = 1 : size(trial_times,1)  % guess the word with SVM
   RorC_number = subject.Train.train1(10,trial_times(i));
   vectors(1,RorC_number) = vectors(1,RorC_number) + 1;
   if predicted_Y_SVM(i) == 1
       vectors(2,RorC_number) = vectors(2,RorC_number) + 1;
   end
   if sum(vectors(1,:)) == 15*12
       current_column = 1;
       current_row = 7;
       target_number_max_row = vectors(2,1);
       target_number_max_column = vectors(2,7);
       for j = 2 : 6
           if vectors(2,j) > target_number_max_column
               target_number_max_column = vectors(2,j);
               current_column = j;
           end
       end
       for j = 8 : 12
           if vectors(2,j) > target_number_max_row
               target_number_max_row = vectors(2,j);
               current_row = j;
           end
       end
       letters = [letters;current_row current_column];
       letter_counter = letter_counter + 1;
       vectors = zeros(2,12);
   end
end

%%  functions

function y=Freqband(x,f1,f2,fs)
L=length(x);
dF = fs/L;
f  = (-fs/2:dF:fs/2-dF)';

if isempty(f1) || f1==-Inf
    BPF = (abs(f) < f2);
elseif isempty(f2) || f2==Inf
    BPF = (f1 < abs(f));
else
    BPF = ((f1 < abs(f)) & (abs(f) < f2));
end
Y=fftshift(fft(x));
P2 = abs(Y);
spec=P2.*BPF;
a=spec.*conj(spec);
y=(1/L)*(sum(a(:)));
end


function gd = groupdelay(h,N)
n = [0:length(h)-1];
jh_prime = n .* h;
gd = real(fft(jh_prime,N) ./ fft(h,N));
end


function y = zphasefilter(h,x)
N = 8000;
yy = filter(h,1,x);
gd = round(groupdelay(h,N));
gd = gd(abs(gd) < 10000);
gd = gd(~isnan(gd));
gd = gd(~isinf(gd));
gd_valid = round(mean(gd));
y = yy(gd_valid + 1 : end);
end






