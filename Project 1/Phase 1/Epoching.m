%
function y=epoching (o,Back_s,Forward_s,h)

l=length(o);
p=length(h);
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