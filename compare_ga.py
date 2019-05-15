import visdom
import sys
import subprocess
import numpy as np

env = 'hopper'
arch = 'rnn'
viz = visdom.Visdom()
plots = {}

deterministic = True
timesteps = 1.5e6
stds  = [0.5]
mrs   = [0.07]
types = ['BASELINE', 'SAFE', 'MOMENTUM', 'SAFE_MOMENTUM', 'AGGRESSIVE']
pool_sizes = [50, 100, 200]

crossovers = [False]

def plot(var_name, split_name, title_name, x, y):
  if title_name not in plots:
    plots[title_name] = viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=env+arch, opts=dict(
        legend=[split_name],
        title=title_name,
        xlabel='Samples',
        ylabel=var_name,
        ytickmin=0,
        ytickmax=1000,
        xtickmin=0,
        xtickmax=timesteps,
        height=500,
        width=1000
    ))
  else:
    viz.line(X=np.array([x]), Y=np.array([y]), env=(env+arch), win=plots[title_name], name=split_name, update = 'append')

def execute(cmd):
  popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
  popen.stdout.flush()
  for stdout_line in iter(popen.stdout.readline, ""):
    yield stdout_line
  popen.stdout.close()
  return_code = popen.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, cmd)


cmd_base = './sieknet ga --train --new ./model/'
counter = 0
for crossover in crossovers:
  for std in stds:
    for mr in mrs:
      for pool_size in pool_sizes:
        for trial in range(16, 20):
          for mutation_type in types:
            counter += 1
            print(viz.get_window_data(env))
            total_experiments = len(stds) * len(mrs) * len(types) * 10 * 2
            print("executing experiment ", counter, " of ", total_experiments)
            cmd_str = cmd_base + \
                      mutation_type + "_" + \
                      env + "_std" + str(std) + \
                      "_mr" + str(mr) \
                      + "_t" + str(trial) + \
                      "." + arch + " " + \
                      " --std "+ str(std) + \
                      " --mr " + str(mr) + \
                      " --mutation_type " + mutation_type + \
                      " --seed " + str(trial) + \
                      " --timesteps " + str(timesteps) + \
                      " --pool_size " + str(pool_size) + \
                      " --" + arch + \
                      " -v"
            if not deterministic:
              cmd_str += " --threads 4"

            if crossover:
              cmd_str += " --crossover"
            for path in execute(cmd_str.split()):
              tokens = path.split()
              if len(tokens) > 0 and tokens[0] == 'Finished!':
                continue
              if len(tokens) != 0 and tokens[0] != 'WARNING:':
                try:
                  gen = int(tokens[1])
                  samples = int(tokens[2])
                  peak = float(tokens[3])
                  avg = float(tokens[4])
                  test = float(tokens[5])

                  plotwin = 'Fixed seed ' + str(trial) + ' std ' + str(std) + ' mr ' + str(mr) + ' crossover ' + str(crossover) + ' p_size ' + str(pool_size) 

                  plot('fitness', mutation_type, plotwin, samples, peak)
                except ValueError:
                  continue

