%Q1.1
clc
clear all
set(gcf,'color','w')
oldprefs=sympref('HeavisideAtOrigin',1);
subplot(4,9,[1,2,3,4,10,11,12,13])
syms t x z
fplot(sin(t),[-pi,pi],'blue','linewidth',1.5)
hold on
fplot(cos(t)*heaviside(t),[-pi,pi],'--','linewidth',1.5);
hold on
W=-pi:pi/70:pi;
plot(W,sin(W)+2*cos(2*W),'go','linewidth',1.5);
%xlim([-pi 4])
%ylim([-3.4 2.4])
%xlabel('t(s)');
ylabel('$y(V)$','interpreter','latex');
title('$$y_{i}(t)$$ for $$i \in \{1,2,3\}$$ on the same figure  ','interpreter','latex');
legend('y_{1}(t)','y_{2}(t)','y_{3}(t)');
hold off
grid on
grid minor
axis([-pi,4,-3.5,2.5])
subplot(4,9,[6,7,8,9])

X=-10:1:20

    Y=(1/2).^X.*heaviside(X);

stem(X,Y,'-b','linewidth',1.5);
grid on
grid minor
p=title('$h[n]=(\frac{1}{2})^nu[n]$','Interpreter','latex')
set(p,'FontSize',12);
%title('$\frac{a}{b}$','Interpreter','latex')
ylabel('$h[n]$','interpreter','latex')
xlabel('$n$','interpreter','latex')

subplot(4,9,[19,20,21,22])
x = linspace(0,10,50);
y = sin(x);
plot(x,y)

txt = '\leftarrow sin(\pi) = 0';
text(pi,sin(pi),txt);
title('\rm \fontsize{8}t(s)');
subplot('position',[0.63 0.33 0.22 0.38]);
circle(4,3,2);
hold on
circle(3,3,1)
hold on
circle(5,3,1)
hold on
circle(4,2,1)
hold on
circle(4,4,1)
grid on

subplot(4,9,[28,29,30,31,32,33,34,35,36])
%figure
P=double(solve(z^2-1.8*cos(pi/16)*z+0.81==0,z));
T=-40:40;
R=(P(1,1)).^T.*heaviside(T);
plot(R,'*r')
hold on
Q=(P(2,1)).^T.*heaviside(T);

plot(Q,'*b')
ht = title('$a_{1}[n]$         $and$       $a_{2}[n]$','interpreter','latex');
set(ht,'FontSize',12);
ylabel('\fontsize{8}Im \{a[n]\}')
xlabel('\fontsize{8}Re \{a[n]\}')
%%
%%Q1.2
clc
clear all
VOTE(50,4,6000)
%%
%Q2.1
clc
clear 
syms z
oldprefs=sympref('HeavisideAtOrigin',1);
%1)

X1=tf([1],[1 -0.8],1,'variable','z^-1')
X2= tf([1],[1 -0.2],1,'variable','z^-1');
figure()
subplot(3,4,[1,2])
v1=1:1:30
i1=impulse(X1);
stem(v1,i1(v1))
title('Impulse Response X1');
subplot(3,4,[5,6,9,10])
pzmap(X1)

subplot(3,4,[3,4])
v2=1:1:20
i2=impulse(X2);
stem(v2,i2(v2))
title('Impulse Response X2');
subplot(3,4,[7,8,11,12])
pzmap(X2)


%2)
X3= tf([1],[1 -1.2],1,'variable','z^-1');
X4= tf([1],[1 -4.8],1,'variable','z^-1');

figure()
subplot(3,4,[1,2])
v3=1:1:20
i3=impulse(X3);
stem(v3,i3(v3))

title('Impulse Response X3');
%xlim([350 390])
subplot(3,4,[5,6,9,10])
pzmap(X3)

subplot(3,4,[3,4])
v4=40:1:60
i4=impulse(X4);
stem(v4,i4(v4))
title('Impulse Response X4');
%xlim([140 180])
subplot(3,4,[7,8,11,12])
pzmap(X4)

%3)
X5= tf([1],[1 -1],1,'variable','z^-1');
X6= tf([1],[1 -2 1],1,'variable','z^-1');

figure()
subplot(3,4,[1,2])
v5=1:1:20
i5=impulse(X5);
stem(v5,i5(v5),'filled')
title('Impulse Response X5');

subplot(3,4,[5,6,9,10])
pzmap(X5)

subplot(3,4,[3,4])
v6=1:1:20
i6=impulse(X6);
stem(v6,i6(v6))
title('Impulse Response X6');
subplot(3,4,[7,8,11,12])
pzmap(X6)

%4)
X7= tf([1 0.5],[1 -1 0.5],1,'variable','z^-1');
X8= tf([1 1],[1 -2 2],1,'variable','z^-1');

figure()
subplot(3,4,[1,2])
v7=0:1:24
i7=impulse(X7);
stem(i7)
title('Impulse Response X7');

subplot(3,4,[5,6,9,10])
pzmap(X7)

subplot(3,4,[3,4])
v8=145:1:160
i8=impulse(X8);
stem(v8,i8(v8))
title('Impulse Response X8');
%xlim([180 200])
subplot(3,4,[7,8,11,12])
pzmap(X8)

%%
%Q2.2

clc
clear
syms n z
oldprefs=sympref('HeavisideAtOrigin',1);
u(n)=heaviside(n);
%1)
x(n)=cos(n*pi/4);
X=ztrans(x);
[n1,d1]=numden(X);
N1=coeffs(n1,[z],'all');
D1=coeffs(d1,[z],'all');
X_=tf(double(N1),double(D1),1,'variable','z^-1');
figure()
pzmap(X_);

%2)
x1=n*x;
X1=ztrans(x1);
X1(z)=simplify(collect(X1));
q=simplify(collect(-z*diff(X)))
%3)
X(z)=X;
X2=X(z^2);


[n2,d2]=numden(X2)
N2=coeffs(n2,[z],'all');
D2=coeffs(d2,[z],'all');
X_2=tf(double(N2),double(D2),1,'variable','z^-1');
[r2,p2,k]=residuez(double(N2),double(D2));
x2(n)=sum((p2.^n).*(r2))*u(n);

figure()
pzmap(X_2);

figure()
subplot(1,2,1)
c=0:1:30
stem(c,x(c),'blue')
xlabel('t')
ylabel('x')
title('x[n]')
subplot(1,2,2)
stem(c,x2(c),'blue')

xlabel('t')
ylabel('x2')
title('x2[n]')
%%
%Q2.3
clc
clear
syms n 
oldprefs=sympref('HeavisideAtOrigin',1);
u(n)=heaviside(n)
%1)
H=tf([1 0.4*sqrt(2)],[1 -0.8*sqrt(2) 0.64],1,'variable','z^-1')
figure()
pzmap(H);
%2)

[N,D]=tfdata(H,'v');
[r,p,k]=residuez(N,D);
h(n)=sum((p.^n).*(r))*u(n);
%3)
%H1=poly2sym(N)/poly2sym(D); %%iztrans is not correct%%
%h1(n)=iztrans(H1);          %%for AntiCausal systems%%
figure()
subplot(1,4,[1 2])
x=0:1:30
stem(h(x),'red')
xlabel('t')
ylabel('h')
title('h[n] with partial fractions')
%xlim([0 35])
subplot(1,4,[3 4])
x=0:1:30
stem(h1(x),'red')
xlabel('t')
ylabel('h')
title('h1[n] with iztrans')
%xlim([0 35])

%4)
h2(n)=sum((p.^n).*(r));
figure()
o=-20:1:20
stem(o,h2(o),'red')
%fplot(h2)
ylabel('h2');
xlabel('t')
title('h2[n] AntiCausal')
%%
%Q2.4

clc
clear
syms n z 
oldprefs=sympref('HeavisideAtOrigin',1);
%1)
H=tf([1 0.5],[1 -1.8*cos(pi/16) 0.81],1,'variable','z^-1');
[N,D]=tfdata(H,'v');
[r,p,k]=residuez(N,D);
h(n)=sum((p.^n).*(r))*heaviside(n);

figure()
x=0:1:40
stem(h(x),'black','filled')
xlim([0 45])
xlabel('t')
ylabel('y')
title('y[n] sec.1')


%2)
%h(0)=1
%h(1)=0.5+1.8cos(pi/16);
B(1,1)=1;
B(2,1)=0.5+1.8*cos(pi/16);
A=vertcat(transpose(p.^(0)),transpose(p.^(1)));
X=linsolve(A,B);
h1(n)=sum((p.^n).*(X))*heaviside(n);
figure()
x=0:1:40
stem(h1(x),'black','filled')
xlabel('t')
ylabel('y')
title('y[n] sec.2')
xlim([0 45])
%3)
b=[1 0.5];
a=[1 -1.8*cos(pi/16) 0.81];
%w=linspace(0,60,61);

t=heaviside(x)-heaviside(x-1);
%e=double(t(w));
y=filter(b,a,t);
figure()
stem(x,y,'black','filled')
xlabel('t')
ylabel('y')
title('y[n] sec.3')
%4)Extra

syms y(n) z
assume(n>=0 & in(n,'integer'))

oldprefs=sympref('HeavisideAtOrigin',1);
u(n)=heaviside(n);
o(n)=u(n)-u(n-1);
f =y(n)-1.8*(cos(pi/16))*y(n-1)+0.81*y(n-2)-o(n)-(1/2)*o(n-1);
fZT =ztrans(f)
syms yZT
fZT = subs(fZT,ztrans(y(n)),yZT)
yZT = solve(fZT,yZT)
ySol = iztrans(yZT);
ySol = simplify(ySol)
ySol = subs(ySol,[y(-1) y(-2)],[0 0])
q=simplify(ySol);
q(n)=q;
figure()
stem(x,q(x),'black','filled')
xlabel('t')
ylabel('y')
title('y[n] sec.4')
%%
%FUNCTIONS 

function h = circle(x,y,r)
th = 0:pi/50:2*pi;
x_ = r * cos(th) + x;
y_ = r * sin(th) + y;
h = plot(x_, y_,'linewidth',1.5);

end

function I=VOTE(n,k,N)
B=ceil(rand(n).*3);
A=B;
%k=625;
%n=50;
v = 1;
q=N;
h = figure();
while ( v<=7000 && isvalid(h))
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