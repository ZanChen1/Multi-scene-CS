
function pip_imshow(I,x_min, y_min)

imshow(I,[]);
set(gca,'Position',[0 0 1 1]);
set (gcf,'Position',[200,100,300,300], 'color','w')%%[x min,y min,x width,y width]
hold on
axes('Position',[0.65,0,0.35,0.35])
axis off
%PiP=I(76:76+32,115:115+32);
PiP=I(x_min:x_min+32,y_min:y_min+32);
PiP(1,1:end)=0;
PiP(end,1:end)=0;
PiP(1:end,1)=0;
PiP(1:end,end)=0;
imshow(PiP,[]);


end