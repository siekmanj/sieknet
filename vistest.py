import visdom
import sys
import subprocess
import numpy as np

viz = visdom.Visdom()
env = "hopper"
plots = {}

def plot(var_name, split_name, title_name, x, y):
  if var_name not in plots:
    plots[var_name] = viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=env, opts=dict(
        legend=[split_name],
        title=title_name,
        xlabel='Generations',
        ylabel=var_name
    ))
  else:
    viz.line(X=np.array([x]), Y=np.array([y]), env=env, win=plots[var_name], name=split_name, update = 'append')

def execute(cmd):
  popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
  popen.stdout.flush()
  for stdout_line in iter(popen.stdout.readline, ""):
    yield stdout_line
  popen.stdout.close()
  return_code = popen.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, cmd)

# Example
print("LOGGING DATA TO VISDOM!")
print("executing ", sys.argv[1:])
for path in execute(sys.argv[1:]):
  tokens = path.split()
  print("got", tokens)
  if len(tokens) != 0 and tokens[0] != 'WARNING:':
    try:
      #print("attempting to convert token 1 to int", tokens[1])
      int(tokens[1])
      #print("attempting to convert token 2 to float", tokens[2])
      float(tokens[2])
      plot('fitness', 'peak', tokens[0], int(tokens[1]), float(tokens[2]))
      plot('fitness', 'mean', tokens[0], int(tokens[1]), float(tokens[3]))
      plot('fitness', 'test', tokens[0], int(tokens[1]), float(tokens[4]))
    except ValueError:
      print("got value error")
      continue
      
print("DONE LOGGING.")
