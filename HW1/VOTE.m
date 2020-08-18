
function I=VOTE(n,k,N)
B=ceil(rand(n).*3);
A=B;
%k=625;
%n=50;
v = 1;
h = figure();
while ( v<=500 && isvalid(h))
for i=1:(n*n/k)
    
    [m,g]=find(A==A(randi(numel(A))));
N=horzcat(m,g);
q=N(randi(size(N,1)),:);
r=q(1,1);
c=q(1,2);
    
    if(r==1 && c==1)
        P=[r+1 c;r+1 c+1;r c+1];
        y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
    elseif(r==1 && c==n)
P=[r c-1;r+1 c-1;r+1 c];
        y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
 elseif(r==n && c==1)
P=[r c+1;r-1 c+1;r-1 c];
        y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
elseif(r==n && c==n)
P=[r c-1;r-1 c-1;r-1 c];
        y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
    elseif(r==1 && (c~=1 && c~=n))
        P=[r c-1;r+1 c-1;r+1 c;r+1 c+1;r c+1];
         y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
         elseif(c==1 && (r~=1 && r~=n))
        P=[r-1 c;r-1 c+1;r c+1;r+1 c+1;r+1 c];
         y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
         elseif(r==n && (c~=n && c~=1))
        P=[r c-1;r-1 c-1;r-1 c;r-1 c+1;r c+1];
         y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
         elseif(c==n && (r~=1 && r~=n))
        P=[r-1 c;r-1 c-1;r c-1;r+1 c-1;r+1 c];
         y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
    elseif((r~=1 && r~=n) && (c~=1 && c~=n))
        P=[r-1 c-1;r-1 c;r-1 c+1;r c+1;r+1 c+1;r+1 c;r+1 c-1;r c-1];
         y=P(randi(size(P,1)),:);
        r_neighbour=y(1,1); c_neighbour=y(1,2);
        A(r,c)=A(r_neighbour,c_neighbour);
        
    end
end
output = zeros(50,50,3);
for i=1:50
    for j=1:50
        if A (i,j) == 1
            output(i,j,:) = [0 1 0];
        end
        if A (i,j) == 2
            output(i,j,:) = [1 0 0];
        end
        if A (i,j) == 3
            output(i,j,:) = [0 0 1];
        end
    end
end
image(output);
title("N = "+v)
pause(0.001);
v = v+1;
end
end