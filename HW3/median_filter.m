function f=median_filter(A1,b)



if mod(b,2)==1
   b=b+1; 
end

k=floor(b/2);



r1=size(A1,1);
r2=size(A1,2);

B=zeros(r1+2*k,r2+2*k);

B(k+1:k+r1,k+1:k+r2)=A1;

for i=k+1:k+r1
    for j=k+1:k+r2
        
        
        c(:,:)=B(i-2:i+2,j-2:j+2);
        
       a=median(c(:));
        
       f(i-k,j-k)=a;

        
        
    end
end

f=mat2gray(f);

end
