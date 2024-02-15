%%
clc
clear all
close all


% Specify the actual lengths in micrometers
actualLengthX = 2400;
actualLengthY = 1800;

% Specify the file paths
filePaths = {'TrackedData.mat'};

% Custom legend labels
legendLabels = {'ALD Alumina', 'MLD Alucone', 'Bare Zn'};

% Cell array to store clotboundary data for each file
allClotboundary = cell(1, numel(filePaths));

% Cell array to store mean absolute deviations for each file and each cell
allMeanAbsoluteDeviations = cell(1, numel(filePaths));

% Cell array to store mean values for each cell's dataset
allMeanYValues = cell(1, numel(filePaths));

% Cell array to store individual thresholds for each cell's dataset
allThresholds = cell(1, numel(filePaths));

% Loop through each file path
for f = 1:numel(filePaths)
    % Load the .mat file
    data = load(filePaths{f});

    % Identify the field containing clotboundary data
    fieldnamesData = fieldnames(data);
    clotboundaryField = findClotboundaryField(data, fieldnamesData);

    % Access the clotboundary variable
    if ~isempty(clotboundaryField)
        allClotboundary{f} = data.(clotboundaryField);

        % Calculate and store mean absolute deviations for each cell
        meanAbsoluteDeviations = zeros(1, numel(allClotboundary{f}));

        for i = 1:numel(allClotboundary{f})
            if ~isempty(allClotboundary{f}{i}) && iscell(allClotboundary{f}{i}) && ~isempty(allClotboundary{f}{i}{1})
                coordinates = allClotboundary{f}{i}{1};
                flippedCoordinates = coordinates(:, [2, 1]);

                % Check if there are valid coordinates
                if size(flippedCoordinates, 1) > 1
                    % Calculate the mean of flippedCoordinates(:, 2)
                    meanYValues(i) = mean(flippedCoordinates(:, 2));

                    % Calculate the threshold (e.g., 30% higher than the mean)
                    thresholds(i) = 1.3 * meanYValues(i);

                    % Filter coordinates within the specified x and y range
                    validIndices = flippedCoordinates(:, 1) >= 1 & flippedCoordinates(:, 1) <= 2400 & flippedCoordinates(:, 2) <= 1.2 * thresholds(i);
                    filteredCoordinates = flippedCoordinates(validIndices, :);

                    % Check if there are valid coordinates after filtering
                    if ~isempty(filteredCoordinates)
                        % Calculate mean absolute deviation
                        meanAbsoluteDeviations(i) = mean(abs(filteredCoordinates(:, 2) - min(filteredCoordinates(:, 2))));
                    else
                        disp(['File ' num2str(f) ', Cell ' num2str(i) ' has no valid coordinates after filtering']);
                        meanAbsoluteDeviations(i) = NaN;
                    end
                else
                    disp(['File ' num2str(f) ', Cell ' num2str(i) ' has insufficient coordinates']);
                    meanAbsoluteDeviations(i) = NaN;
                end
            else
                disp(['File ' num2str(f) ', Cell ' num2str(i) ' is empty or does not contain valid coordinates']);
                meanAbsoluteDeviations(i) = NaN;
            end
        end
        allMeanAbsoluteDeviations{f} = meanAbsoluteDeviations;
        allMeanYValues{f} = meanYValues;
        allThresholds{f} = thresholds;
    else
        disp(['File ' num2str(f) ' - Clotboundary variable not found or is empty']);
    end
end

% Display mean absolute deviations, mean values, and thresholds for each cell and file
for f = 1:numel(filePaths)
    disp(['File ' num2str(f) ' Mean Absolute Deviations:']);
    disp(allMeanAbsoluteDeviations{f});

    disp(['File ' num2str(f) ' Mean Y Values:']);
    disp(allMeanYValues{f});

    disp(['File ' num2str(f) ' Thresholds:']);
    disp(allThresholds{f});
end

% Create the folder if it doesn't exist
outputFolder = 'Tracked_surface_compare_plots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Plot all cell data for each file in a single plot
for i = 1:numel(allClotboundary{1}) % Assuming all files have the same number of cells
    % Check if the variable exists and is not empty
    if ~isempty(allClotboundary{1}{i})
        % Plot each set of coordinates from all files
        figure;
        hold on;

        for f = 1:numel(filePaths)
            clotboundary = allClotboundary{f};

            % Check if the variable exists and is not empty
            if iscell(clotboundary{i}) && ~isempty(clotboundary{i}{1})
                coordinates = clotboundary{i}{1};  % Assuming the coordinates are in the first cell

                % Swap x and y coordinates
                flippedCoordinates = coordinates(:, [2, 1]);
                % Calculate the mean of flippedCoordinates(:, 2)
                meanY = mean(flippedCoordinates(:, 2));
                % Calculate the threshold for filtering (30% higher than the mean)
                threshold = allThresholds{f}(i);

                % Filter coordinates within the specified x and y range
                filteredCoordinates = flippedCoordinates(flippedCoordinates(:, 1) >= 0 & flippedCoordinates(:, 1) <= 2400 & flippedCoordinates(:, 2) <= 0.74 * threshold, :);

                % Check if there are valid coordinates after filtering
                if ~isempty(filteredCoordinates)
                    % Rescale the x and y coordinates
                    scaledX = (filteredCoordinates(:, 1) / 1440) * actualLengthX; %max x value in rawdata file
                    scaledY = (filteredCoordinates(:, 2) / 1080) * actualLengthY; %max x value in rawdata file

                    % Plot the filtered coordinates for each file
                    scatter(scaledX, scaledY, '.', 'DisplayName', legendLabels{f});
                

                    title([' ' num2str((i-1)*10) ' min'],FontWeight="normal");
                    xlabel('X (\mum)');
                    ylabel('Y (\mum)');
                    box on;
                    % Set custom x-axis tick marks with a gap of 600
                    customXTicks = 0:600:2400;
                    % Set x-axis ticks
                    set(gca, 'XTick', customXTicks);
                   
                   % Reverse the y-axis
                    axis(gca, 'ij');

                    % Set y-axis limits
                    ylim([1100 1600]);
                    xlim([0 2400]);
                    
                    % Save the figure with the plot title as the filename
                    figName = fullfile(outputFolder, [' ' num2str((i-1)*10) ' min' '.jpg']);
                    saveas(gcf, figName);
                else
                    disp(['File ' num2str(f) ', Cell ' num2str(i) ' has no valid coordinates after filtering']);
                end
            else
                disp(['File ' num2str(f) ', Cell ' num2str(i) ' is empty or does not contain valid coordinates']);
            end
        end

        %legend('show');
        box on;
        hold off;
    else
        disp(['Cell ' num2str(i) ' data is empty or not found']);
    end
end


% Plot mean absolute deviations for each cell and file with connecting lines
figure;
hold on;

for f = 1:numel(filePaths)
    meanAbsoluteDeviations = allMeanAbsoluteDeviations{f};

    % Check if the variable exists and is not empty
    if ~isempty(meanAbsoluteDeviations)
        % Plot mean absolute deviations for each cell with connecting lines
        plot((0:(numel(meanAbsoluteDeviations)-1)) * 10, meanAbsoluteDeviations, '-o', 'DisplayName', legendLabels{f});
    else
        disp(['File ' num2str(f) ' Mean Absolute Deviations is empty or not found']);
    end
end

title('Mean Absolute Deviations for Each Cell and File');
xlabel('Time (min)');
ylabel('Mean Absolute Deviation');
legend('show');
box on;
hold off;

% Create the folder if it doesn't exist
outputFolder = 'Tracked_surface_compare_plots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save the figure
figName = fullfile(outputFolder, 'Mean_Absolute_Deviation_plot.jpg');
saveas(gcf, figName);




% Find the maximum absolute deviation across all datasets
maxAbsDeviation = max(cellfun(@max, cellfun(@abs, allMeanAbsoluteDeviations, 'UniformOutput', false)))

% Plot normalized mean absolute deviations for each cell and file with connecting lines
figure;
hold on;

for f = 1:numel(filePaths)
    meanAbsoluteDeviations = allMeanAbsoluteDeviations{f};

    % Check if the variable exists and is not empty
    if ~isempty(meanAbsoluteDeviations)
        % Normalize mean absolute deviations with respect to the largest absolute deviation
        normalizedDeviations = meanAbsoluteDeviations / maxAbsDeviation;

        % Plot normalized mean absolute deviations for each cell with connecting lines
        plot((0:(numel(normalizedDeviations)-1)) * 10, normalizedDeviations, '-o', 'DisplayName', legendLabels{f}, 'LineWidth', 1.5, 'MarkerFaceColor', 'auto');
        ytickformat('%.1f');
    else
        disp(['File ' num2str(f) ' Mean Absolute Deviations is empty or not found']);
    end
end

%title('Normalized Mean Absolute Deviations for Each Cell and File');
xlabel('Time (min)');
ylabel('Normalized Surface Roughness');
legend('show');
box on;
hold off;

% Create the folder if it doesn't exist
outputFolder = 'Tracked_surface_compare_plots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save the figure
figName = fullfile(outputFolder, 'normalized_mean_absolute_deviations_plot.jpg');
saveas(gcf, figName);

% Function to find clotboundary variable in the structure
%%
function clotboundaryField = findClotboundaryField(data, fieldnamesData)
    clotboundaryField = [];

    % Iterate through field names
    for i = 1:numel(fieldnamesData)
        currentField = data.(fieldnamesData{i});

        % Check if the current field is a cell array with non-empty cells
        if iscell(currentField) && ~isempty(currentField) && all(cellfun(@iscell, currentField)) && all(cellfun(@(x) ~isempty(x{1}), currentField))
            clotboundaryField = fieldnamesData{i};
            break;
        end
    end
end