
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
%imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.
%workspace;  % Make sure the workspace panel is showing.
clf;
fontSize = 14;

experiment='19';
test='3';
tube='3';

NAME=append(experiment,'.',test,'.',tube)
% folder=fullfile(append('/Volumes/Matteo Files/Kastrup Internship /FINAL ANALYSIS/',experiment,'/',experiment,'.',test))
folder=fullfile('/Volumes/Matteo/Matteo Contraction Files/19.3/19.3.3')
% workingdirectory = append('/Volumes/Matteo Files/Kastrup Internship /FINAL ANALYSIS/',experiment,'/',experiment,'.',test,'/',NAME)
workingdirectory = '/Volumes/Matteo/Matteo Contraction Files/19.3/19.3.3'
cd(workingdirectory)

% Change the current folder to the folder of this m-file.
% mfilename
% if(~isdeployed)
%   cd(fileparts(which(mfilename)));
% end

% Open the rhino.avi demo movie that ships with MATLAB.
%folder = fullfile(matlabroot, '\toolbox\images\imdemos');

%movieFullFileName = fullfile(folder, 'rhinos.avi');
movieFullFileName = fullfile(folder, append(NAME,'mask.mov'));
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
  npics=50
  frameincrement=round(numberOfFrames/npics);
  numberOfFramesWritten = 0;

  % Prepare a figure to show the images in the upper half of the screen.
%   figure(1);
%  screenSize = get(0, 'ScreenSize');
%  newWindowPosition = [1 screenSize(4)/2 - 70 screenSize(3) screenSize(4)/2];
%   set(gcf, 'Position', newWindowPosition); % Maximize figure.
%   set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
  
%     % Ask user if they want to write the individual frames out to disk.
%     promptMessage = sprintf('Do you want to save the individual frames out to individual disk files?');
%     button = questdlg(promptMessage, 'Save individual frames?', 'Yes', 'No', 'Yes');
%     if strcmp(button, 'Yes')
        writeToDisk = true;
          % Extract out the various parts of the filename.
          [folder, baseFileName, extentions] = fileparts(movieFullFileName);
          % Make up a special new output subfolder for all the separate
          % movie frames that we're going to extract and save to disk.
          % (Don't worry - windows can handle forward slashes in the folder name.)
          folder = pwd;   % Make it a subfolder of the folder where this m-file lives.
          outputFolder = sprintf('%s/Movie Frames from %s', folder, baseFileName);
          % Create the folder if it doesn't exist already.
          if ~exist(outputFolder, 'dir')
              mkdir(outputFolder);
          end
%       else
%           writeToDisk = false;        
%       end
      % Loop through the movie, writing all frames out.
    % Each frame will be in a separate file with unique name.
        ii=1;
        for frame = 1 : frameincrement : frameincrement*npics

            ThisFrame = read(movieInfo,frame);
            outputBaseFileName2 = sprintf('0%02d.png', ii);
            outputFullFileName2 = fullfile(outputFolder, outputBaseFileName2);
            imwrite(ThisFrame,outputFullFileName2);

            I = imread(outputFullFileName2);
            IMAGETEXT = sprintf('0%02d', ii);
            ii=ii+1;
            RGB = insertText(I,[10 201],IMAGETEXT,FontSize=10,TextColor="red");
            imwrite(RGB,outputFullFileName2)
        end


        cols=10;
        rows=npics/cols;

        duration=movieInfo.Duration; % video sped up duration[s]
        speedup=100;
        actualtime=duration*speedup;
        realtime=actualtime/npics;

%         mkdir tracked
        clotboundary=cell(1,npics);
        %Analyze each picture (frame)
        for ii=1:npics
            PICFileName = sprintf('0%02d.png', ii);
            PICFullFileName = fullfile(outputFolder, PICFileName);
            filename = compose('0%02d.png', ii); %name of file as outputted by VideoProc Software
            Icolor = imread(PICFullFileName); %Import image





            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Define thresholds for bright red regions

            redThreshold = 26; % Adjust this threshold as needed for your images 65 50 40 26 179 159
            greenThreshold = 230; % Adjust this threshold as needed for your images 70 75 31 77 68 70
            blueThreshold = 230; % Adjust this threshold as needed for your images 70 75 51 179 51 100
            sered = strel('disk', 10); % Adjust the size of the disk as needed red mask (tubes)
            
            % Create a mask for bright red regions
            redMask = Icolor(:, :, 1) > redThreshold & Icolor(:, :, 2) < greenThreshold & Icolor(:, :, 3) < blueThreshold;

            % Convert red mask to grayscale for area calculation
            redGray = uint8(redMask) * 255;
            redGray=imcomplement(redGray);
            % Apply morphological operations for smoothing
            redGray = imdilate(redGray, sered);
            redGray = imerode(redGray, sered);

            % Find connected components (individual regions) in the binary image
            cc = bwconncomp(redGray);

            % Measure properties of connected components
            props = regionprops(cc, 'Area');

            % Get areas and indices of regions sorted in descending order
            areas = [props.Area];
            [~, sortedIndices] = sort(areas, 'descend');

            % Keep only the three largest areas and remove others
            for i = 4:numel(sortedIndices)
                redGray(cc.PixelIdxList{sortedIndices(i)}) = 0;
            end

            % Store areas for each tube in this frame
            gelwoundarea(ii) = props(sortedIndices(1)).Area;

            % Find perimeter (boundary) of the red region in the binary image
            perim = bwperim(redGray,18);
            % Create a copy of the original image to draw contours
            f2 = figure(22);
            trackedImg = Icolor;
            % Draw contours on the tracked image for visualization
            trackedImg(perim) = 255; % Set the boundary pixels to 0 (black) in the original image
            imshow(trackedImg);

            [boundariesredmask,l,n,AA] = bwboundaries(redGray);
            % Find all regions (big+small) outlined by the bwboundaries function
            Aredmask=zeros(1,n);
            mredmask=cell(1,n);
            for k=1:n
                bredmask = boundariesredmask{k};
                [mredmask{k},Aredmask(k)] = boundary(bredmask(:,1),bredmask(:,2));
                areaaredmask(k)=polyarea(bredmask(:,1),bredmask(:,2));
            end
            areathreshold=20000;
            indices = find(areaaredmask<areathreshold);
            areaaredmask(indices) = [];
            indicesA = find(Aredmask<areathreshold);
            Aredmask(indicesA) = [];
            indicesAA = find(Aredmask>8000000);
            Aredmask(indicesAA) = [];
            Areassredmask(ii,:)=[areaaredmask(1:1),Aredmask(1:1)]
            boundariesredmask(indicesA)=[];
            clotboundaryredmask{ii}=boundariesredmask{1};

            filenametracked = compose('0%02d_tracked.png', ii); %name of file as outputted by VideoProc Software
            foldernametracked = append(pwd,'/trackedredmaskRED'); %name of file as outputted by VideoProc Software
            if ~exist(foldernametracked, 'dir')
                mkdir('trackedredmaskRED');
            end
            set(gca,'LooseInset',get(gca,'TightInset'));
            F= getframe(f2);
            imgf2 = F.cdata;
            imwrite(imgf2, fullfile(foldernametracked, filenametracked{1}));
            imageArrayRedMask{ii}=imread(fullfile(foldernametracked, filenametracked{1}));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            imageArraycolor{ii}=Icolor;
            Igrey = rgb2gray(Icolor); %Convert to grayscale
            imageArraygreyscale{ii}=Igrey;
            T = graythresh(Igrey)-0.06
            BW = imbinarize(Igrey,T);
            BW=BW(:,:,1);
            BW=imcomplement(BW);
            imageArraybinary{ii}=BW;
            dim = size(BW);
            row = round(dim(1)/2);
            col = 50;
            BW_filled = imfill(BW,'holes');
            imageArrayBB_filled{ii}=BW_filled;
            [boundaries,l,n,AA] = bwboundaries(BW_filled);
            % Find all regions (big+small) outlined by the bwboundaries function
            A=zeros(1,n);
            m=cell(1,n);
            for k=1:n
                b = boundaries{k};
                [m{k},A(k)] = boundary(b(:,1),b(:,2));
                areaa(k)=polyarea(b(:,1),b(:,2));
            end


            %%%%%%%%%%%%%%
            areathreshold=20000;
            indices = find(areaa<areathreshold);
            areaa(indices) = [];
            indicesA = find(A<areathreshold);
            A(indicesA) = [];
            indicesAA = find(A>8000000);
            A(indicesAA) = [];
            Areass(ii,:)=[areaa(1:1),A(1:1)]
            boundaries(indicesA)=[];
            clotboundary{ii}=boundaries{1};
            %%%%%%%%%%%%%%

            %Plot the two regions in green overlayed on the image
            f=figure(102);
            Icontour=Icolor;
            imshow(Icontour);
            hold on
            for k=1:1
                b = boundaries{k};
                bredmask = boundariesredmask{k};
                plot(b(:,2),b(:,1),'g','LineWidth',3);
                plot(bredmask(:,2),bredmask(:,1),'b','LineWidth',3)
            end
            filenametracked = compose('0%02d_tracked.png', ii); %name of file as outputted by VideoProc Software
            foldernametracked = append(pwd,'/tracked'); %name of file as outputted by VideoProc Software
            if ~exist(foldernametracked, 'dir')
                mkdir('tracked');
            end
            set(gca,'LooseInset',get(gca,'TightInset'));
            F= getframe(f);
            img102 = F.cdata;
            imwrite(img102, fullfile(foldernametracked, filenametracked{1}));
            imageArraycontour{ii}=imread(fullfile(foldernametracked, filenametracked{1}));

            %Plot the two regions in green overlayed on the image
            f=figure(103);
            Icontour=Icolor;
            imshow(Icontour);
            hold on
            for k=1:1
                %         b = boundaries{k};
                bredmask = boundariesredmask{k};
                %         plot(b(:,2),b(:,1),'g','LineWidth',3);
                plot(bredmask(:,2),bredmask(:,1),'b','LineWidth',3)
            end
            filenametracked = compose('0%02d_tracked.png', ii); %name of file as outputted by VideoProc Software
            foldernametracked = append(pwd,'/trackedredmask'); %name of file as outputted by VideoProc Software
            if ~exist(foldernametracked, 'dir')
                mkdir('trackedredmask');
            end
            set(gca,'LooseInset',get(gca,'TightInset'));
            F= getframe(f);
            img103 = F.cdata;
            imwrite(img103, fullfile(foldernametracked, filenametracked{1}));
            imageArraycontourredmask{ii}=imread(fullfile(foldernametracked, filenametracked{1}));
        end


        %Pixel areas graphing (primary checkpoint)
        figure(10)
%         plot(linspace(1,npics,npics).*realtime./60,Areass(:,2),'linewidth',4)
        hold on
%         plot(linspace(1,npics,npics).*realtime./60,Areass(:,1),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,gelwoundarea,'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,Areassredmask(:,2),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,Areassredmask(:,1),'linewidth',4)
        xlabel('Time [min]')
        ylabel('Clot Projected Area [pixels]')
        title('Evolution of Clot Projected Area in Time')
        legend('Clot Projected AreaBW1','Clot Projected AreaBW2','Clot Projected Area Redmask','Clot Projected Area Redmask1','Clot Projected Area Redmask2')
        print(gcf, 'ContractionCurve.png', '-dpng', '-r500');  % 300 DPI


        %Pixel areas graphing (primary checkpoint)
        figure(11)
%         plot(linspace(1,npics,npics).*realtime./60,Areass(:,2)./max(Areass(:,2)),'linewidth',4)
        hold on
%         plot(linspace(1,npics,npics).*realtime./60,Areass(:,1)./max(Areass(:,1)),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,gelwoundarea./max(gelwoundarea),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,Areassredmask(:,2)./max(Areassredmask(:,2)),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,Areassredmask(:,1)./max(Areassredmask(:,1)),'linewidth',4)
        xlabel('Time [min]')
        ylabel('Normalized Clot Projected Area (Max) [-]')
        title('Evolution of Clot Projected Area in Time')
        legend('Clot Projected AreaBW1','Clot Projected AreaBW2','Clot Projected Area Redmask','Clot Projected Area Redmask1','Clot Projected Area Redmask2')
        print(gcf, 'ContractionCurveNormalizedMax.png', '-dpng', '-r500');  % 300 DPI

        %Pixel areas graphing (primary checkpoint)
        figure(12)
        plot(linspace(1,npics,npics).*realtime./60,Areass(:,2)./Areass(1,2),'linewidth',4)
        hold on
        plot(linspace(1,npics,npics).*realtime./60,Areass(:,1)./Areass(1,1),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,gelwoundarea./gelwoundarea(1),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,Areassredmask(:,2)./Areassredmask(1,2),'linewidth',4)
        plot(linspace(1,npics,npics).*realtime./60,Areassredmask(:,1)./Areassredmask(1,1),'linewidth',4)
        xlabel('Time [min]')
        ylabel('Normalized Clot Projected Area (Initial) [-]')
        title('Evolution of Clot Projected Area in Time')
        legend('Clot Projected AreaBW1','Clot Projected AreaBW2','Clot Projected Area Redmask','Clot Projected Area Redmask1','Clot Projected Area Redmask2')
        print(gcf, 'ContractionCurveNormalizedInitial.png', '-dpng', '-r500');  % 300 DPI


        % Ask user if they want to write the individual frames out to disk.
        promptMessage = sprintf('Do you want to save the montages?');
        button = questdlg(promptMessage, 'Save All Montages (takes minutes)?', 'Yes', 'No', 'Yes');
        if strcmp(button, 'Yes')
            figure (13)
            hhhh=montage(imageArraycolor, 'size', [rows cols])
            title('Initial Images')
            set(gcf,'Visible','on')
            savefig('Wounds.fig')
            montage_IM=hhhh.CData;
            ImFileOut = fullfile(pwd, 'Wounds.png');
            imwrite(montage_IM,ImFileOut);
            % saveas(gcf, 'Wounds2.png', 'png', 'Resolution', 1000);
            print(gcf, 'Wounds.png', '-dpng', '-r3000');  % 300 DPI

            figure(14)
            hhhhh=montage(imageArrayBB_filled, 'size', [rows cols])
            set(gcf,'Visible','on')
            savefig('BWFilled.fig')
            montage_IM=hhhhh.CData;
            ImFileOut = fullfile(pwd, 'BWFilled.png');
            imwrite(montage_IM,ImFileOut);
            print(gcf, 'BWFilled.png', '-dpng', '-r3000');  % 300 DPI

            figure(15)
            hhhhhh=montage(imageArraycontour, 'size', [rows cols])
            set(gcf,'Visible','on')
            savefig('TrackedMontage.fig')
            montage_IM=hhhhhh.CData;
            ImFileOut = fullfile(pwd, 'TrackedMontage.png');
            imwrite(montage_IM,ImFileOut);
            print(gcf, 'TrackedMontage.png', '-dpng', '-r4000');  % 300 DPI

            figure(16)
            hhhhhhh=montage(imageArrayRedMask, 'size', [rows cols])
            set(gcf,'Visible','on')
            savefig('TrackedMontage.fig')
            montage_IM=hhhhhhh.CData;
            ImFileOut = fullfile(pwd, 'TrackedMontageRedmask.png');
            imwrite(montage_IM,ImFileOut);
            print(gcf, 'TrackedMontageRedMask.png', '-dpng', '-r3000');  % 300 DPI

            NormalizedAreas=Areass(:,1)./Areass(1,1);
            Time=linspace(1,npics,npics).*realtime./60;
            %Need to save all the data to files in folder for ease of getting back.
            % TimeSI=linspace(1,npics,npics).*realtime./60;
            fprintf(fopen('AreaBW1.txt','w'),'%.5f ',Areass(:,1));
            fprintf(fopen('AreaBW2.txt','w'),'%.5f ',Areass(:,2));
            fprintf(fopen('Arearedmask.txt','w'),'%.5f ',gelwoundarea);
            fprintf(fopen('Arearedmask1.txt','w'),'%.5f ',Areassredmask(:,1));
            fprintf(fopen('Arearedmask2.txt','w'),'%.5f ',Areassredmask(:,2));
            fprintf(fopen('NormalizedAreaBW.txt','w'),'%.5f ',Areass(:,1)./Areass(1,1));
            fprintf(fopen('NormalizedArearedmask.txt','w'),'%.5f ',gelwoundarea./gelwoundarea(1));
            fprintf(fopen('TimeSI.txt','w'),'%.5f ',linspace(1,npics,npics).*realtime./60);

            save("TrackedData.mat","clotboundary","clotboundaryredmask","Areass","Areassredmask","gelwoundarea","Igrey","T","imageArraybinary","imageArrayBB_filled","imageArraycontour","imageArraycolor","NormalizedAreas","Time")

        else
            figure (13)
            hhhh=montage(imageArraycolor, 'size', [rows cols])
            title('Initial Images')
            set(gcf,'Visible','on')
            set(gcf, 'Position', [100, 100, 2000, 1000]); % Set figure position and size

            figure(15)
            hhhhhh=montage(imageArraycontour, 'size', [rows cols])
            set(gcf,'Visible','on')
            set(gcf, 'Position', [100, 100, 2000, 1000]); % Set figure position and size


        end
