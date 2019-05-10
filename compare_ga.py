import visdom
import sys
import subprocess
import numpy as np

env = "hopper"
viz = visdom.Visdom()
plots = {}

def plot(var_name, split_name, title_name, x, y):
  if title_name not in plots:
    plots[title_name] = viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=env, opts=dict(
        legend=[split_name],
        title=title_name,
        xlabel='Samples',
        ylabel=var_name
    ))
  else:
    viz.line(X=np.array([x]), Y=np.array([y]), env=env, win=plots[title_name], name=split_name, update = 'append')

def execute(cmd):
  popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
  popen.stdout.flush()
  for stdout_line in iter(popen.stdout.readline, ""):
    yield stdout_line
  popen.stdout.close()
  return_code = popen.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, cmd)

timesteps = 4e6
arch = 'mlp'
stds  = [0.25, 0.5, 1.0]
mrs   = [0.01, 0.05, 0.1]
types = ['BASELINE', 'SAFE', 'MOMENTUM', 'SAFE_MOMENTUM']

cmd_base = './sieknet ga --train --new ./model/'
counter = 0
for trial in range(10):
  for crossover in [True, False]:
    for std in stds:
      for mr in mrs:
        for mutation_type in types:
          counter += 1
          total_experiments = len(stds) * len(mrs) * len(types) * 10 * 2
          print("executing experiment ", counter, " of ", total_experiments, ": ", cmd_str)
          cmd_str = cmd_base + \
                    mutation_type + "_" + \
                    env + "_std" + str(std) + \
                    "_mr" + str(mr) \
                    + "_t" + str(trial) + \
                    ".mlp --std " + str(std) + \
                    " --mr " + str(mr) + \
                    " --mutation_type " + mutation_type + \
                    " --seed " + str(trial) + \
                    " --timesteps " + \
                    " --" + arch + \
                    str(timesteps) + \
                    " -v"

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

                plotwin = 'Fixed seed ' + str(trial) + ' std ' + str(std) + ' mr ' + str(mr) + ' crossover ' + str(crossover)

                plot('fitness', mutation_type + ' peak', plotwin, samples, peak)
                plot('fitness', mutation_type + ' mean', plotwin, samples, avg)

              except ValueError:
                continue

