config = dict(
    dataset='Cifar10',
    model='resnet18_nobias',
    optimizer='SGD',
    optimizer_decay_at_epochs=[150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=256,
    num_epochs=300,
    seed=42,
)

# if __name__ == '__main__':
#     import sys
#     import os
#     sys.path.append('..')
#     import train

#     train.output_dir = os.path.join('output', os.path.basename(__file__))
#     os.makedirs(train.output_dir)
#     train.config = config
#     best_accuracy = train.main()
#     with open(os.path.join(train.output_dir, 'best_accuracy.txt'), 'w') as fp:
#         fp.write(str(best_accuracy))
