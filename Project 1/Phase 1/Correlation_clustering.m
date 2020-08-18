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
