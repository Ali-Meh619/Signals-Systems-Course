%%
%Q1
%1)
clc 
clear
syms s t
u(t)=heaviside(t);
f1_s=laplace(5*exp(-5*t)*u(t))
f2_s=laplace(5*t*exp(-5*t)*u(t))
f3_s=laplace((t*sin(2*t)+exp(-2*t))*u(t))
f4_s=laplace(5*(t^2)*exp(-5*t)*u(t))

pretty(f1_s)
pretty(f2_s)
pretty(f3_s)
pretty(f4_s)
%2)
F1_t=ilaplace(28/(s*(s+8)))
F2_t=ilaplace(s-5/(s*((s+2)^2)))
F3_t=ilaplace(10/((s+1)^2*(s+3)))
F4_t=ilaplace(2*(s+1)/(s*(s^2+s+2)))

pretty(F1_t)
pretty(F2_t)
pretty(F3_t)
pretty(F4_t)

%3)
G=tf([0 0 25],[1 4 25])
figure();
impulseplot(G)
figure
stepplot(G)
figure()
stepplot(G)
bode(G)
%%
%Q.2
%1)
clc
clear
syms t s
u(t)=heaviside(t);

H1=tf([0 0 0 1],[1 20 10 400])
H2=tf([0 0 0 0 1],[1 12.5 10 10 1])
H3=tf([0 0 0 0 0 0 1],[1 5 125 100 100 20 10])
H4=tf([0 0 0 0 0 1],[1 125 100 100 20 10])

H_i=[H1 H2 H3 H4];
j=[1 2 3 4]
figure_control(j,H_i)

p_1=pole(H1)
p_2=pole(H2)
p_3=pole(H3)
p_4=pole(H4)

%2)
figure
subplot(2,2,1)
stepplot(H1)
title('Step Response H_{1}')
subplot(2,2,2)
stepplot(H2)
title('Step Response H_{2}')
subplot(2,2,3)
stepplot(H3)
title('Step Response H_{3}')
subplot(2,2,4)
stepplot(H4)
title('Step Response H_{4}')
%4)

G1=tf([0 1 1],[1 4 4])
figure
subplot(2,2,1)
impulseplot(G1)
subplot(2,2,2)
stepplot(G1)
x3(t)=sin(2*t)*u(t)
p=0:0.001:5
o=double(x3(p))
subplot(2,2,3)
lsim(G1,o,p)
title('Response to \itsin(2t)u(t)')
legend('Response')
x4(t)=exp(-t)*u(t)
p=0:0.001:5
q=double(x4(p))
subplot(2,2,4)
lsim(G1,q,p)
title('Response to \ite^{-t}u(t)')
legend('Response')
%5)

G2=tf([0 10 4],[1 4 4])
G_2=(10*s+4)/(s^2+4*s+4)
St_R=ilaplace((1/s)*G_2)
figure
stepplot(G2)
[e,w]=step(G2)
lsiminfo(e,w,1)
S1 =stepinfo(G2,'RiseTimeThreshold',[0 0.5])
st1 = S1.RiseTime
%%
%Q.3
%1)
clc
clear
syms s t K

H=tf([0 0 1],[1 1 -2])
figure
pzmap(H)

%2)

M=feedback(H,1)
figure
pzmap(M)

%C =tf([0],[1]);
%C.InputName = 'e';
%C.OutputName = 'u';
%H.InputName = 'u';
%H.OutputName = 'Y';
%Sum = sumblk('e = X - Y');
%T = connect(H,C,Sum,'X','Y','u')

%N=(k)/(s^2 +s+k-2)
%U=subs(N,k,-10:2:10)

k=-10:2:10;
for i=1:11
    Q(1,i)=tf([0 0 k(1,i)],[1 1 -2+k(1,i)]);
    
end
figure_control_2(k)

%4)
y=-30:1:30
figure
rlocus(H,y)

%5)
b=K/(s^2 +s-2+K)
f=poles(b,s)
pretty(f)
%%





%%
%%FUNCTIONS
function figure_control(fs,H)
f = figure;
c = uicontrol(f,'Style','popupmenu');
c.Position = [1 1 120 20];
c.String = strsplit(num2str(fs));
c.Callback = @selection;
    function selection(src,event)
        val = c.Value;
        str = c.String;
        F = str2double(str{val});
        
 p=H(1,F);
 pzmap(p)
 if isstable(p)==1               
                title("system is stable  H"+F);
 else
     title("system is not stable  H"+F);
                
 end
    end
end

function figure_control_2(fs)
f = figure;
c = uicontrol(f,'Style','popupmenu');
c.Position = [1 1 120 20];
c.String = strsplit(num2str(fs));
c.Callback = @selection;
    function selection(src,event)
        val = c.Value;
        str = c.String;
        F = str2double(str{val});
        
 p=tf([0 0 F],[1 1 F-2]);
 
 pzmap(p)
 if isstable(p)==1               
                title("System is stable for k= "+F);
 else
     title("System is not stable for k= "+F);
                
 end
    end
end