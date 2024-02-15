    clc;    % Clear the command window.
    close all;  % Close all figures (except those of imtool.)
    clear;  % Erase all existing variables.
    fontSize = 14;
    
    %folder=fullfile('D:\Experiment\In_situ_OM\package_development_Musanna_Matteo\Tracking_Bare_Zn');
    %cd 'D:\Experiment\In_situ_OM\package_development_Musanna_Matteo\Tracking_Bare_Zn'
    movieFileName = 'movie.mp4';
    movieFullFileName = fullfile(fileparts(mfilename('fullpath')), movieFileName);
        
    % Check to see that it exists.
    if ~exist(movieFullFileName, 'file')
      strErrorMessage = sprintf('File not found:\n%s\nYou can choose a new one, or cancel', movieFullFileName);
      response = questdlg(strErrorMessage, 'File not found', 'OK - choose a new movie.', 'Cancel', 'OK - choose a new movie.');
      if strcmpi(response, 'OK - choose a new movie.')
        [baseFileName, folderName, FilterIndex] = uigetfile('*.avi');
        if ~isequal(baseFileName, 0)
          movieFullFileName = fullfile(folderName, baseFileName);
        else
          return;
        end
      else
        return;
      end
    end
    
    movieInfo = VideoReader(movieFullFileName);
    videoheight=movieInfo.Height;
    videowidth=movieInfo.Width;
    % Determine how many frames there are.
    numberOfFrames = movieInfo.NumFrames;

% Read npics from the text file
fid = fopen('variables.txt', 'r');
npics = fscanf(fid, '%d');
fclose(fid);

% Now npics variable contains the value passed from Python
disp(['Value of npics received from Python: ' num2str(npics)]);


    frameincrement=round(numberOfFrames/npics);
    numberOfFramesWritten = 0;
    duration=movieInfo.Duration; % video sped up duration[s]
    frames = round(logspace(0,log10(numberOfFrames),npics));
    speedup=100;
    actualduration=duration*speedup;
    framerate=actualduration/numberOfFrames;
    TimeActual = frames.*framerate;
    
    % Ask user if they want to write the individual frames out to disk.
    %     promptMessage = sprintf('Do you want to save the individual frames out to individual disk files?');
    %     button = questdlg(promptMessage, 'Save individual frames?', 'Yes', 'No', 'Yes');
    button='Yes';
    if strcmp(button, 'Yes')
        writeToDisk = true;
        % Extract out the various parts of the filename.
        [folder, baseFileName, extentions] = fileparts(movieFullFileName);
        folder = pwd;   % Make it a subfolder of the folder where this m-file lives.
        outputFolder = sprintf('%s/Movie Frames from %s', folder, baseFileName);
        % Create the folder if it doesn't exist already.
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end
    else
        writeToDisk = false;
    end
    % Loop through the movie, writing all frames out.
    % Each frame will be in a separate file with unique name.
    ii=1;
    for frame = 1 : frameincrement : frameincrement*npics
    
        % Extract the frame from the movie structure.
        %thisFrame = mov(frame).cdata;
        %       thisFrame = read(movieInfo,frame);
        %       imshow(thisFrame)
    
        ThisFrame = read(movieInfo,frame);
        outputBaseFileName2 = sprintf('0%02d.png', ii);
        outputFullFileName2 = fullfile(outputFolder, outputBaseFileName2);
        imwrite(ThisFrame,outputFullFileName2);
    
        I = imread(outputFullFileName2);
        IMAGETEXT = sprintf('0%02d', ii);
        ii=ii+1;
        RGB = insertText(I,[10 201],IMAGETEXT,FontSize=10,TextColor="red");
        imwrite(RGB,outputFullFileName2);
    end
    mkdir tracked
    clotboundary=cell(1,npics);
    clotarea=zeros(1,npics);
    T=zeros(1,npics);
    
    for iii=1:npics
        PICFileName = sprintf('0%02d.png', iii);
        PICFullFileName = fullfile(outputFolder, PICFileName);
        filename = compose('0%02d.png', iii); %name of file
        Icolor = imread(PICFullFileName); %Import image
        imageArraycolor{iii}=Icolor;
    
        button2='N';
        thresholdadd=0.0;
        while button2 == 'N'
            [clotboundary{iii},clotarea(iii),T(iii)]=trackimageefficient(Icolor,thresholdadd);
            [img] = plottrackedimageoverlay(Icolor,clotboundary{iii},iii);
            clf
            figure(iii)
            clf
            imshow(img)
         
            set(gcf,'Visible','on')
            promptMessage = sprintf('Is area tracked well?');
            button2 = questdlg(promptMessage, 'Is clot tracked well?', 'Y', 'N', 'Y');
            if strcmp(button2, 'N')
                thresholdadd=thresholdadd+0.02;
            end
        end
    
        %Analyze picture frame iii for final time to get all info
        [clotboundary{iii},clotarea(iii),Igrey{iii},T(iii),imageArraybinary{iii},imageArrayBB_filled{iii}] = trackimagefull(Icolor,thresholdadd);
        [img] = plottrackedimageoverlay(Icolor,clotboundary{iii},iii);
        filenametracked = compose('0%02d_tracked.png', iii); %name of file as outputted by VideoProc Software
        foldernametracked = append(pwd,'/tracked'); %name of file as outputted by VideoProc Software
        imwrite(img, fullfile(foldernametracked, filenametracked{1}));
        imageArraycontour{iii}=imread(fullfile(foldernametracked, filenametracked{1}));
    end
    
    
    %Actual Area (SI Units)
    %AreasSI(:,1)=Areass(:,2).*A_barSI./Areass(:,1);
    %AreasSI(:,1)=Areass(:,1);
    Areas(:,1)=clotarea;
    %AreasSI(:,2)=Areass(:,4).*A_barSI./Areass(:,3);
    %AreasSI(:,2)=Areass(:,2);
    
    duration=movieInfo.Duration; % video sped up duration[s]
    speedup=100;
    actualtime=duration*speedup;
    realtime=actualtime/npics;
    
    cols=npics;
    rows=npics/cols;
    fignumbers=[iii+1,iii+2,iii+3,iii+4,iii+5];
    hold off
    
    
    figure(fignumbers(1))
    plot(linspace(1,npics,npics).*realtime./60,Areas(:,1),'linewidth',4)
    % hold on
    % plot(linspace(1,npics,npics).*realtime./60,AreasSI(:,2),'linewidth',4)
    xlabel('Time [min]')
    ylabel('Clot Cross-Sectional Area [pixel]')
    title('Evolution of Clot Projected Area in Time')
    savefig('ContractionPixel.fig')
    % legend('polyhedra area','boundary area')
    
    figure (fignumbers(2))
    hhhh=montage(imageArraycolor, 'size', [rows cols]);
    title('Initial Images')
    montage_IM=hhhh.CData;
    ImFileOut = fullfile(pwd, 'WoundMontage.png');
    imwrite(montage_IM,ImFileOut);
    savefig('WoundsMontage.fig')
    
    % % figure(14)
    % % montage(imageArraygreyscale, 'size', [rows cols])
    % % figure(15)
    % % montage(imageArraybinary, 'size', [rows cols])
    
    figure(fignumbers(3))
    hhhhh=montage(imageArrayBB_filled, 'size', [rows cols]);
    montage_IM=hhhhh.CData;
    ImFileOut = fullfile(pwd, 'BWTrackeddMontage.png');
    imwrite(montage_IM,ImFileOut);
    savefig('BWTrackedAreasMontage.fig')
    cla() 
    figure(fignumbers(4))
    hhhhhh=montage(imageArraycontour, 'size', [rows cols]);
    montage_IM=hhhhhh.CData;
    ImFileOut = fullfile(pwd, 'TrackedMontage.png');
    imwrite(montage_IM,ImFileOut);
    savefig('TrackedAreasMontage.fig')
    
    %%%% Re-edit frames in case of mistake
    button3 = questdlg('Would you like to re-edit a frame?', 'Would you like to re-edit a frame?', 'Y', 'N', 'Y');
    while button3 == 'Y'
        prompt = {'Enter frame number:'};
        dlgtitle = 'Input';
        fieldsize = [1 45];
        definput = {'1'};
        frame = inputdlg(prompt,dlgtitle,fieldsize,definput);
        iii=str2num(frame{1});
        PICFileName = sprintf('0%02d.png', iii);
        PICFullFileName = fullfile(outputFolder, PICFileName);
    %     filename = compose('0%02d.png', iii); %name of file
        Icolor = imread(PICFullFileName); %Import image
        imageArraycolor{iii}=Icolor;
    
        button2='N';
        thresholdadd=0.0;
        while button2 == 'N'
            [clotboundary{iii},clotarea(iii),T(iii)]=trackimageefficient(Icolor,thresholdadd);
            [img] = plottrackedimageoverlay(Icolor,clotboundary{iii},iii);
            clf
            figure(iii)
            clf
            imshow(img)
    
            set(gcf,'Visible','on')
            promptMessage = sprintf('Is area tracked well?');
            button2 = questdlg(promptMessage, 'Is clot tracked well?', 'Y', 'N', 'Y');
            if strcmp(button2, 'N')
                thresholdadd=thresholdadd+0.02;
            end
        end
    
        %Analyze picture frame iii for final time to get all info
        [clotboundary{iii},clotarea(iii),Igrey{iii},T(iii),imageArraybinary{iii},imageArrayBB_filled{iii}] = trackimagefull(Icolor,thresholdadd);
        [img] = plottrackedimageoverlay(Icolor,clotboundary{iii},iii);
        filenametracked = compose('0%02d_tracked.png', iii); %name of file as outputted by VideoProc Software
        foldernametracked = append(pwd,'/tracked'); %name of file as outputted by VideoProc Software
        imwrite(img, fullfile(foldernametracked, filenametracked{1}));
        imageArraycontour{iii}=imread(fullfile(foldernametracked, filenametracked{1}));
        
        Areas(:,1)=clotarea;
        iii=npics;
        fignumbers=[iii+1,iii+2,iii+3,iii+4,iii+5];
    
        delete(figure(fignumbers(1)))
        figure(fignumbers(1))
        clf
        plot(linspace(1,npics,npics).*realtime./60,Areas(:,1),'linewidth',4)
        % hold on
        % plot(linspace(1,npics,npics).*realtime./60,AreasSI(:,2),'linewidth',4)
        xlabel('Time [min]')
        ylabel('Clot Cross-Sectional Area [pixel]')
        title('Evolution of Clot Projected Area in Time')
        set(gcf,'Visible','on')
    %     legend('polyhedra area','boundary area')
    
        figure(fignumbers(2))
        clf
        montage(imageArraycolor, 'size', [rows cols])
        title('Initial Images')
    
        % % figure(14)
        % % montage(imageArraygreyscale, 'size', [rows cols])
    
        % % figure(15)
        % % montage(imageArraybinary, 'size', [rows cols])
    
        figure(fignumbers(3))
        clf
        montage(imageArrayBB_filled, 'size', [rows cols])
    
        figure(fignumbers(4))
        clf
        montage(imageArraycontour, 'size', [rows cols])
    
        button3 = questdlg('Would you like to re-edit a frame?', 'Would you like to re-edit a frame?', 'Y', 'N', 'Y');
    end
    
    
    delete(figure(fignumbers(1)))
    figure(fignumbers(1))
    clf
    NormalizedAreas=Areas(:,1)./max(Areas(:,1));
    plot(linspace(1,npics,npics).*realtime./60,Areas(:,1)./max(Areas(:,1)),'linewidth',4)
    % hold on
    % plot(linspace(1,npics,npics).*realtime./60,AreasSI(:,2),'linewidth',4)
    xlabel('Time [min]')
    ylabel('Normalized Anode Cross-Sectional Area')
    title('Evolution of Stripping/Plating Projected Area in Time')
    set(gcf,'Visible','on')
    savefig('NormalizedContractionPlot.fig')
    %     legend('polyhedra area','boundary area')
    
    %Need to save all the data to files in folder for ease of getting back.
    % TimeSI=linspace(1,npics,npics).*realtime./60;
    fprintf(fopen('Area.txt','w'),'%.5f ',Areas);
    fprintf(fopen('NormalizedArea.txt','w'),'%.5f ',NormalizedAreas);
    fprintf(fopen('TimeSI.txt','w'),'%.5f ',TimeActual);
    
    save("TrackedData.mat","clotboundary","clotarea","Igrey","T","imageArraybinary","imageArrayBB_filled","imageArraycontour","Areas","imageArraycolor","NormalizedAreas")
    
    function [clotboundary,area,Igrey,T,BW,BW_filled] = trackimagefull(pic,thresholdadd)
        Igrey = rgb2gray(pic); %Convert to grayscale
        T = graythresh(Igrey)-thresholdadd;
        BW = imbinarize(Igrey,T);
        BW=BW(:,:,1);
        BW=imcomplement(BW);
        BW_filled = imfill(BW,'holes');
        [boundaries,l,n,AA] = bwboundaries(BW_filled);
        % Find all regions (big+small) outlined by the bwboundaries function
        A=zeros(1,n);
        m=cell(1,n);
        for k=1:n
            b = boundaries{k};
            [m{k},A(k)] = boundary(b(:,1),b(:,2));
            areaa(k)=polyarea(b(:,1),b(:,2));
        end
        %Find biggest area region (clot)
        [areaa,I]=sort(areaa,'descend');
        area=areaa(1);
        clotboundary={boundaries{I(1)}};
    end
    
    %%
    function [clotboundary,areaclot,T] = trackimageefficient(pic,thresholdadd)
        Igrey = rgb2gray(pic); %Convert to grayscale
    %     figure(5)
    %     imshow(Igrey)
        T = graythresh(Igrey)-thresholdadd;
        BW = imbinarize(Igrey,T);
    %     figure(6)
    %     imshow(BW)
        BW=BW(:,:,1);
        BW=imcomplement(BW);
    %     figure(6)
    %     imshow(BW)
        BW_filled = imfill(BW,'holes');
    %     figure(8)
    %     imshow(BW_filled)
        [boundaries,l,n,AA] = bwboundaries(BW_filled);
        % Find all regions (big+small) outlined by the bwboundaries function
        A=zeros(1,n);
        m=cell(1,n);
        for k=1:n
            b = boundaries{k};
            [m{k},A(k)] = boundary(b(:,1),b(:,2));
            areaa(k)=polyarea(b(:,1),b(:,2));
        end
        %Find biggest area region (clot)
        [areaa,I]=sort(areaa,'descend');
        areaclot=areaa(1);
        clotboundary={boundaries{I(1)}};
    end
    
    function [img] = plottrackedimageoverlay(original,boundary,fignum)
    
        %Plot the two regions in green overlayed on the image
        Icontour=original;
        Ifinal=figure(fignum);
        clf
        imshow(Icontour);
        hold on
        b = boundary{1};
        plot(b(:,2),b(:,1),'g','LineWidth',3);
    
    %     filenametracked = compose('0%02d_tracked.png', ii); %name of file as outputted by VideoProc Software
    %     foldernametracked = append(pwd,'/tracked'); %name of file as outputted by VideoProc Software
        set(gca,'LooseInset',get(gca,'TightInset'));
        f = figure(fignum);
        F= getframe(f);
        img = F.cdata;
    
    %     imwrite(img, fullfile(foldernametracked, filenametracked{1}));
    %     figure(1)
    %     imageArraycontour{ii}=imread(fullfile(foldernametracked, filenametracked{1}));
    
    end
