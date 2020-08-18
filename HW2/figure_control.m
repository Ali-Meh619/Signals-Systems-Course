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