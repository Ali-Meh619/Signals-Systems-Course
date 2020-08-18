load 64channeldata.mat
load filter_2.mat
ww=size(data,2);
Y=fft(data(1,:,1));
fs=600;
P2 = abs(Y/ww);
P1 = P2(1:ww/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(ww/2))/ww;

plot(f,P1) ;
title('Single-Sided Amplitude Spectrum of X(t)');
xlabel('f (Hz)');
ylabel('|P1(f)|');

%%
NN=size(data,3);
ww=size(data,2);
vv=size(data,1);
qq=zeros(vv,ww,NN);


bb=zeros(63,63,NN);


for jj=1:NN
    
for i=1:vv

qq(i,:,jj)=filter(filter_2,data(i,:,jj));

end
end
  



%%
ff=zeros(63,7200,11);


for k=1:11
    
   ff(:,:,k)=horzcat(qq(:,:,4*k-3),horzcat(qq(:,:,4*k-2),horzcat(qq(:,:,4*k-1),qq(:,:,4*k) ) ) );

end
%%
R=zeros(63,63,11);

for jj=1:11
for ii=1:vv

    for j=1:vv
        
        R(ii,j,jj)=corr(ff(ii,:,jj),ff(j,:,jj));
        
    end
  end 
end

for jj=1:11
rr(:,:,jj)=R(:,:,jj)-diag(diag(R(:,:,jj)));
end
%%

Matrix   = rr;
 Method   = 'UPGMA';
  
  Parameter='Limit';
  Value=0.3;
  field='Cluster';

  
  
  
  for jj=1:11
[x y z]=Correlation_clustering(Matrix(:,:,jj),Method,Parameter,Value);


value(jj,:)={y};

  end
Cluster=struct(field,value);

%%

r1=size(Epoch,1);
r2=size(Epoch,2);
r3=size(Epoch,3);
eee=zeros(9,256,9);

i=0;
for jj=1:9
i=i+1;
while(mod(i,300)~= 0)

eee(:,:,jj)=eee(:,:,jj)+Epoch(:,:,i);

i=i+1;



end

eee(:,:,jj)=eee(:,:,jj)/300;
end
%%



for jj=1:9
for ii=2:9

    for j=2:9
        
        R1(ii-1,j-1,jj)=corr(eee(ii,:,jj),eee(j,:,jj));
        
    end
  end 
end

for jj=1:9
rrr(:,:,jj)=R1(:,:,jj)-diag(diag(R1(:,:,jj)));
end

%%



Matrix   = rrr;
 Method1   = 'UPGMA';
  
  
  
  Parameter1='Limit';
  Value1=0.25;
  field1='Cluster1';

  
  
  
  for jj=1:9
[x y1 z]=Correlation_clustering(Matrix(:,:,jj),Method,Parameter,Value);


value1(jj,:)={y1};

  end
Cluster1=struct(field1,value1);



%%




function y = bpfilt(signal, f1, f2, fs, isplot)

if nargin < 4 || isempty(fs)
	fs = 1;
end
if nargin < 5 || isempty(isplot)
	isplot = 1;
end

if isrow(signal)
    signal = signal';
end
N  = length(signal);
dF = fs/N;
f  = (-fs/2:dF:fs/2-dF)';

if isempty(f1) || f1==-Inf
    BPF = (abs(f) < f2);
elseif isempty(f2) || f2==Inf
    BPF = (f1 < abs(f));
else
    BPF = ((f1 < abs(f)) & (abs(f) < f2));
end
signal 	 = signal-mean(signal);

spektrum = fftshift(fft(signal))/N;
if isplot
    
    subplot(2,1,1);
    plot(f,abs(spektrum));
    title('Power spectrum of the original signal');
end

spektrum = BPF.*spektrum;
if isplot
    subplot(2,1,2);
    plot(f,abs(spektrum));
    title(sprintf('filtered signal in (%.3f, %.3f) Hz',f1,f2));
end

y = ifft(ifftshift(spektrum)); 
y = abs(y);
if isplot
    time = 1/fs*(0:N-1)';
	
    
    subplot(2,1,1);
    plot(time,signal);
    title('The original time series');
    subplot(2,1,2);
    plot(time,y);
    title(sprintf('The band-pass filtered time series in (%.3f, %.3f) Hz',f1,f2));
end
end



function y=epoching (o,Back_s,Forward_s,h)



l=length(o);
p=length(h)


m1=o(1,:);
m2=o(2,:);
T=m1(1,2)-m1(1,1);
sb=floor(Back_s/T);
sf=floor(Forward_s/T);
EPOCH=zeros(9,sb+sf+1,p);
for i=1:p
                
    EPOCH(:,:,i) = o(1:9,h(1,i)-sb:h(1,i)+sf);   
        
end
y=EPOCH;
end

function v=corr(x,y)
a=x.*y;
a1=sum(a(:));
b=x.*x;
b1=sum(b(:));
c=y.*y;
c1=sum(c(:));
v=(a1)/sqrt(c1*b1);
end


function M=CorrelationCluster (x,y)

x=x-triu(x);

d=1-abs(x)


end


function varargout = Correlation_clustering(Matrix,Method,Parameter,Value)

Matrix = Data_control(Matrix,Method,Parameter,Value);

% Linkage
Tree = Linkage(Matrix,Method);

[Roots, Clusters, Number] = Analysis(Tree,Parameter,Value);


        varargout{1} = Tree;
        varargout{2} = Clusters;
        varargout{3} = Roots;
  
end
% Data control
function Matrix = Data_control(Matrix,Method,Parameter,Value)

[n,m] = size(Matrix);


switch upper(Method)
    case {'WPGMA','UPGMA'}
    otherwise, error('Clustering method must be ''WPGMA'' or ''UPGMA''.');
end


switch lower(Parameter)
    case {'number','limit'}
    otherwise
        error('Clustering parameter must be ''Number'' or ''Limit''.');
end
% Parameter value
if ~isnumeric(Value)
    error('Parameter ''%s'' is not numeric.',Parameter);
end
end

function Tree = Linkage(Matrix,Method)

N = size(Matrix,1);


Md=1-abs(Matrix)


Indices  = 1:N;
Weights  = ones(1,N);
Tree = zeros(N-1,3);
% Linkage
for t = 1:N-1
    
    
    [n,m] = find(Md == min(min(Md)),1,'first');
    
    % New cluster corresponding to the minimum dissimilarity
    Tree(t,:) = [Indices(sort([n m])), Md(n,m)];
    
    
    I = numel(Indices);
    D = zeros(I,1);
    
    % Distance vector wrt new cluster
    for i = 1:I
        
        if i < n,  d1 = Md(n,i);
        else       d1 = Md(i,n);
        end
        
        
        if i <= m, d2 = Md(m,i);
        else       d2 = Md(i,m);
        end
        
        switch upper(Method)
            
            
            case 'WPGMA'
                D(i) = mean([d1 d2]);
                
              
            case 'UPGMA'
                D(i) = (Weights(n)*d1 + Weights(m)*d2) / (Weights(n) + Weights(m)); 
                
        end
        
    end
    
  
    Indices(n) = [];  Indices(m) = [];    
    Indices = [N+t Indices]; 
    
   
    w = Weights(n) + Weights(m);
    Weights(n) = [];  Weights(m) = []; 
    Weights = [w Weights] ; 
    
    
    D(n) = [];        D(m) = [];        
    Md([n m],:) = []; Md(:,[n m]) = [];  
    Md = [inf(1,I-1); D,Md];
    
end
end
% Analysis of tree
function [Roots, Clusters, Number] = Analysis(Tree,Parameter,Value)
% Empty roots and nodes vectors
Roots = [];
Nodes = [];
% Vector length
N = max(max(Tree(:,1:2)))/2+1;
% Clustering parameter
switch lower(Parameter)
    
    % Number of clusters
    case 'number'
        Number = Value;
        Limit = Tree(end-Number+1+1,3);
        if Limit == 0
            return
        end
        
  
    case 'limit'
        Limit = Value;
        Number = N-find(Tree(:,3)>=Limit,1,'first')+1;
        
end
% Clusters
Clusters = cell(Number,1);
% Cluster number
Cluster = 0;
% Exploration of the node
ExplorationDown(Tree(end,1),Tree(end,3));
ExplorationDown(Tree(end,2),Tree(end,3));
    
    function ExplorationDown(Node,Dissimilarity)
        
        if ismember(Node,Nodes) || ismember(Node,Roots)
            return
        end
        
       
        Nodes = [Nodes Node];        
        
        if Node <= N
            
            % Root
            Roots = [Roots Node];
           
            
            [n,~]=find(Tree(:,1:2)==Node);            
            if Tree(n,3) >= Limit
                Cluster = Cluster+1;
            end
            
          
            Clusters{Cluster} = [Clusters{Cluster} Node];
            
        else
            
            
            Node = Node-N;
            
           
            N1 = Tree(Node,1);
            N2 = Tree(Node,2);
            
           
            if Tree(Node,3) < Limit && Dissimilarity >= Limit
                Cluster = Cluster+1;
            end
            
           
            Dissimilarity = Tree(Node,3);
            
            if N1 <= N && N2 <= N
                
              
                if N1 < N2
                    ExplorationDown(N1,Dissimilarity);
                    ExplorationDown(N2,Dissimilarity);
                else
                    ExplorationDown(N2,Dissimilarity);
                    ExplorationDown(N1,Dissimilarity);
                end
                
            else
                
                
                ExplorationDown(N1,Dissimilarity);
                ExplorationDown(N2,Dissimilarity);
                
            end
            
        end
        
    end
end

