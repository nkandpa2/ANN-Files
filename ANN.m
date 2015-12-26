trainingData = [];
for a = 1:50
    x = rand*6 - 3;
    trainingData = [trainingData [x; exp(x) + 4 + (rand - 0.5)]];
end
    
neurons = [];
for a = 1:70
neurons = [neurons [rand*10 - 5; rand*10 - 5]];
end

learningRate = 1;
Cost = [];
dc = 40;
error = 40;

%for b = 1:1000
while (abs(dc) > 0.01 || error > 0.5) && abs(dc) > 0.0005
output = [];
gradient = zeros(2,70);

for a = trainingData
    input = a(1);
    weightedInputs = input * neurons(1,:) + neurons(2,:);
    activations = 1 ./ (1 + exp(-weightedInputs));
    o = sum(activations);
    output = [output o];
    
    gradWeights = (o - a(2))*(exp(-weightedInputs)./(1 + exp(-weightedInputs)).^2)*(input);
    gradBiases = (o - a(2))*(exp(-weightedInputs)./(1 + exp(-weightedInputs)).^2);
    gradient = [gradient + [gradWeights; gradBiases]];
end

error = 1/2 * sum((trainingData(2,:) - output).^2);
Cost = [Cost error];
gradient = gradient / 50;
if(max(size(Cost)) > 1)
    dc = Cost(end) - Cost(end - 1);
end
neurons = neurons - learningRate*gradient;
end


plot(Cost(:,2:end), 'kx')

figure;

plot(trainingData(1,:), trainingData(2,:), 'kx', trainingData(1,:), output, 'bx')
