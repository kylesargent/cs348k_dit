import matplotlib.pyplot as plt
from experiment_helpers import experiments

for i, exp in enumerate(experiments):
    steps, losses = exp.get_training_loss()
    smoothing_window = 10
    smooth_losses = []
    for i in range(0, len(losses) - smoothing_window, smoothing_window):
        smooth_losses.append(sum(losses[i:i+smoothing_window]) / smoothing_window)
    losses = smooth_losses
    steps = steps[:-smoothing_window:smoothing_window]

    plt.plot(steps, losses, label=exp.tag)
    # plt.plot(val_steps, val_losses, label=exp.tag + ' val')
plt.yscale('log')

# axis labels
plt.xlabel('training iterations')
plt.ylabel('loss')

# set yrange
plt.ylim(1.25e-1, 1.5e-1)

plt.legend()
plt.savefig('training_loss.png')


plt.figure()

for i, exp in enumerate(experiments):
    steps, losses = exp.get_validation_loss()
    smoothing_window = 5
    smooth_losses = []
    for i in range(0, len(losses) - smoothing_window, smoothing_window):
        smooth_losses.append(sum(losses[i:i+smoothing_window]) / smoothing_window)
    losses = smooth_losses
    steps = steps[:-smoothing_window:smoothing_window]

    plt.plot(steps, losses, label=exp.tag)
    # plt.plot(val_steps, val_losses, label=exp.tag + ' val')
plt.yscale('log')

# axis labels
plt.xlabel('training iterations')
plt.ylabel('loss')

# set yrange
plt.ylim(1.25e-1, 1.5e-1)

plt.legend()

plt.savefig('validation_loss.png')




for i, exp in enumerate(experiments):
    steps, losses = exp.get_training_loss()
    smoothing_window = 5
    smooth_losses = []
    for i in range(0, len(losses) - smoothing_window, smoothing_window):
        smooth_losses.append(sum(losses[i:i+smoothing_window]) / smoothing_window)
    losses = smooth_losses
    steps = steps[:-smoothing_window:smoothing_window]

    val_steps, val_losses = exp.get_validation_loss()
    smoothing_window = 5
    smooth_losses = []
    for i in range(0, len(val_losses) - smoothing_window, smoothing_window):
        smooth_losses.append(sum(val_losses[i:i+smoothing_window]) / smoothing_window)
    val_losses = smooth_losses
    val_steps = val_steps[:-smoothing_window:smoothing_window]


    plt.plot(steps, losses, label=exp.tag + ' train')
    plt.plot(val_steps, val_losses, label=exp.tag + ' val')
plt.yscale('log')

# axis labels
plt.xlabel('training iterations')
plt.ylabel('loss')

# set yrange
plt.ylim(1.25e-1, 1.5e-1)

# plt.legend()

plt.savefig('both.png')




import matplotlib.pyplot as plt
from cycler import cycler

# Define a set of colors to cycle through
colors = plt.cm.tab10.colors

# Initialize the plot
fig, ax = plt.subplots()

for i, exp in enumerate(experiments):
    steps, losses = exp.get_training_loss()
    smoothing_window = 10
    smooth_losses = []
    for j in range(0, len(losses) - smoothing_window, smoothing_window):
        smooth_losses.append(sum(losses[j:j+smoothing_window]) / smoothing_window)
    losses = smooth_losses
    steps = steps[:-smoothing_window:smoothing_window]

    val_steps, val_losses = exp.get_validation_loss()
    smooth_losses = []
    for j in range(0, len(val_losses) - smoothing_window, smoothing_window):
        smooth_losses.append(sum(val_losses[j:j+smoothing_window]) / smoothing_window)
    val_losses = smooth_losses
    val_steps = val_steps[:-smoothing_window:smoothing_window]

    # Assign the color based on the index
    color = colors[i % len(colors)]
    ax.plot(steps, losses, label=exp.tag + ' train', color=color)
    ax.plot(val_steps, val_losses, label=exp.tag + ' val', color=color, linestyle='--')

ax.set_yscale('log')

# Axis labels
ax.set_xlabel('training iterations')
ax.set_ylabel('loss')

# Set y-range
ax.set_ylim(1.25e-1, 1.5e-1)

# Add legend
# ax.legend()

# Save the plot
plt.savefig('both.png')