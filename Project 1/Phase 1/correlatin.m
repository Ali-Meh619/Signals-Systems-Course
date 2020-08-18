%
function v=corr(x,y)
a=x.*y;
a1=sum(a(:));
b=x.*x;
b1=sum(b(:));
c=y.*y;
c1=sum(c(:));
v=(a1)/sqrt(c1*b1);
end