// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "csv.h"
#include "nn.h"

// Read csv for features
torch::Tensor read_data()
{

    io::CSVReader<4> in("../data/features.csv");
    in.read_header(io::ignore_extra_column, "feature1", "feature2", "feature3", "feature4");
    int feature1;
    int feature2;
    int feature3;
    int feature4;

    std::vector<int> myvector{};
    while (in.read_row(feature1, feature2, feature3, feature4))
    {
        myvector.push_back(feature1);
        myvector.push_back(feature2);
        myvector.push_back(feature3);
        myvector.push_back(feature4);
    }

    torch::Tensor inputs = torch::tensor(myvector).to(torch::kFloat32).view({-1, 4});

    return inputs.clone();
}

// Read csv for labels
torch::Tensor read_label()
{

    io::CSVReader<1> in("../data/labels.csv");
    in.read_header(io::ignore_extra_column, "label");
    int label;

    std::vector<int> myvector{};
    while (in.read_row(label))
    {
        myvector.push_back(label);
    }

    torch::Tensor label_tensor = torch::tensor(myvector).to(torch::kFloat32).view({-1, 1});
    return label_tensor.clone();
}
// Pack features and label tensors
std::pair<torch::Tensor, torch::Tensor> load_dataset()
{
    return std::make_pair(read_data(), read_label());
}

// Custom Dataset class
class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
    /* data */
    // Should be 2 tensors
    torch::Tensor states, labels;
    size_t ds_size;

public:
    CustomDataset(torch::Tensor list_images, torch::Tensor list_labels)
    {
        states = (list_images);
        labels = (list_labels);
        ds_size = states.sizes()[0];
    };

    torch::data::Example<> get(size_t index) override
    {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states[index];
        torch::Tensor sample_label = labels[index];
        return {sample_img.clone(), sample_label.clone()};
    };

    torch::optional<size_t> size() const override
    {
        return ds_size;
    };
};

int main(int argc, const char *argv[])
{
    std::cout << "Logistic Regression\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t input_size = 4;
    const int64_t hidden_size = 2;
    const int64_t num_classes = 1;
    const int64_t batch_size = 8;
    const size_t num_epochs = 10;
    const double learning_rate = 0.001;

    // Get paths of images and labels as int from the folder paths
    std::pair<torch::Tensor, torch::Tensor> pair_images_labels = load_dataset();

    torch::Tensor list_images = pair_images_labels.first;
    torch::Tensor list_labels = pair_images_labels.second;

    // Initialize CustomDataset class and read data
    auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());

    // Data loaders
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset), batch_size);

    // Number of training examples
    auto num_train_samples = 1000;
    // Logistic regression model
    NeuralNet model(input_size, hidden_size, num_classes);

    // Load pretrained model
    torch::load(model, argv[1]);
    model->to(device);

    // Freeze subset of pre-trained model
    int count_param = 0;
    for (auto &parameter : model->parameters())
    {
        count_param += 1;
        if (count_param < 3)
            parameter.requires_grad_(false);
    }

    // Loss and optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *data_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            optimizer.zero_grad();

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::binary_cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.round();

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }
    std::cout << "model after: " << std::endl
              << model->parameters() << std::endl;
    torch::save(model, "../models/model_cpp.pt");
    std::cout << "Training finished!\n\n";
}
