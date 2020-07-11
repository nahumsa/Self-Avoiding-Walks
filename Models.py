import numpy as np

class SAW():
  """ Class for the Self-Avoiding Walk.

  Parameters
  ----------------------------------
  N (int): Size of the latice.
  initial_x (int): Initial x point.
  initial_y (int): Initial y point.
  """
  def __init__(self,N: int, initial_x: int=0, initial_y: int=0):
    self.grid = self._grid(N)
    self.k = []

    # Initial Value
    self.x = [initial_x]
    self.y = [initial_y]
    
    
    self.n_walks = self._evolution()    

  def _evolution(self, plot=False, debug=False):
    """ Makes the evolution of the SAW.

    Parameters 
    -----------------------------------
    plot (bool): True if you want to plot. (default=False)
    debug (bool): True if you want to debug. (default=False)


    Outputs
    -----------------------------------
    (int): number of steps. 


    """
    # Set initial values    
    initial = (self.x[0],self.y[0])
    n_steps = 0
    colide = False

    # Do the first walk    
    possible_walks,possible_values = self._check_walks(self.grid,*initial)  
    
    # Log possible walks
    self.k.append(len(possible_walks))  
    
    direction = self._choose_uniformly(possible_walks)    
    i,j = self._walk(self.grid, *initial, direction, debug=debug)
    self.x.append(i)
    self.y.append(j)
    
    n_steps += 1
    

    while not colide:
      possible_walks, possible_values = self._check_walks(self.grid, i, j)
      
      # Log possible walks
      self.k.append(len(possible_walks))  

      # Check if there is no possible ways to walk 
      # if this is the case then the walker colided and it is terminated
      
      if len(possible_walks) == 0:
        colide = True
        
        # Exclude last value because it is 0
        self.k.pop(-1)
        
        break

      direction = self._choose_uniformly(possible_walks)    
      i,j = self._walk(self.grid, i, j, direction, debug=debug)      
      self.x.append(i)
      self.y.append(j)
      
      n_steps += 1            
      
      if plot:
        plt.imshow(self.grid)
        plt.show()
    
    return n_steps

  @property
  def number_of_walks(self):
    """ Returns the number of walks of the walker.

    """
    return self.n_walks  

  @property
  def trial_probability(self):
    """ Return the log of possible walks.

    """
    g = (1/np.array(self.k)).cumprod()
    return g[-1]

  def _grid(self, N: int):
    """ Creates a NxN grid.

    Parameters 
    -----------------------------------

    N (int): Lenght of the size of the grid.


    Outputs
    -----------------------------------
    (np.array): NxN grid


    """
    return np.zeros((N,N))

  def _choose_uniformly(self, x: list):
    """ Chooses uniformly a random value from an array.

    Parameters
    -----------------------------------
    x (array): Array that you want to choose a value uniformly.


    Output
    -------------------------------------
    (int or float): Value chosen from the array.

    """
    n = len(x)
    index = 0
    if n > 1:
      index = np.random.randint(0,high=n)     
    return x[index]

  def _walk(self, matrix: np.array
               ,i: int,j: int
               ,direction: int, debug=False):
    """ Walk on the grid, here we put a value for the direction:
        1: Left
        2: Right
        3: Up
        4: Down
    

    Parameters
    -----------------------------------
    matrix (np.array): Grid that you wand to walk.
    i (int): Y index of the array.
    j (int): X index of the array.
    direction (int): Direction of the step.
    debug (boolean): True if you want to debug the function.
                     (Default=False).    

    """
    # Check if it passes the lenght of the array
    if i == len(matrix) or j == len(matrix):
      print('Error')
      return None, None
    
    # Check if it is going left
    elif direction == 1:
      if debug:
        print('Left')
      matrix[i,j] = direction
      return i,j-1
    
    # Check if it is going right
    elif direction == 2:
      if debug:
        print('Right')
      matrix[i, j] = direction
      return i,j+1
    
    # Check if it is going up
    elif direction == 3:
      if debug:
        print('Up')
      matrix[i, j] = direction
      return i-1,j

    # Check if it is going down
    elif direction == 4:
      if debug:
        print('Down')
      matrix[i, j] = direction
      return i+1,j

  def _check_walks(self, matrix,i,j):
    """ Check possible walks on position i,j.
    

    Parameters
    -----------------------------------
    matrix (np.array): Grid that you wand to walk.
    i (int): Y index of the array.
    j (int): X index of the array.
    
    Output
    -----------------------------------
    (dict): Dictionary with possible walks, keys are directions
            1,2,3,4 and values are indices associated with each
            direction.

    """
    possible_walks = {1:(i,j-1), 
                      2:(i,j+1), 
                      3:(i-1,j), 
                      4:(i+1,j)}
    # Create a list for all directions that are not possible
    impossible_walks = []

    # See impossible walks
    for key, indexes in zip(possible_walks,possible_walks.values()):
      try:
        if matrix[indexes] != 0 or indexes[0] < 0 or indexes[1] < 0:
          impossible_walks.append(key)
      except:
        impossible_walks.append(key)

    # Delete from the dict all impossible walks
    for delete in impossible_walks:
      possible_walks.pop(delete)
    
    # Return all possible walks and all indices
    return list(possible_walks.keys()), list(possible_walks.values())


class SAWEarly(SAW):
  """ Class for the Self-Avoiding Walk with a 
  early termination probability.

  Parameters
  ----------------------------------
  N (int): Size of the latice.
  initial_x (int): Initial x point.
  initial_y (int): Initial y point.
  terminate_probability(float): Probability that a walk can terminate at each 
                                step.

  """
  def __init__(self, 
               N: int, 
               initial_x: int=0, 
               initial_y: int=0, 
               terminate_probability: float=0.1):
    
    self.terminate_probability = terminate_probability
    super(SAWEarly, self).__init__(N,initial_x,initial_y)
        

  def _evolution(self, plot=False, debug=False):
    """ Makes the evolution of the SAW.

    Parameters 
    -----------------------------------
    plot (bool): True if you want to plot. (default=False)
    debug (bool): True if you want to debug. (default=False)


    Outputs
    -----------------------------------
    (int): number of steps. 


    """
    # Set initial values    
    initial = (self.x[0],self.y[0])
    n_steps = 0
    colide = False

    # Do the first walk    
    possible_walks,possible_values = self._check_walks(self.grid,*initial)  
    

    # Log possible walks
    self.k.append(len(possible_walks))  
    
    if np.random.uniform() < self.terminate_probability:
      colide = True

    direction = self._choose_uniformly(possible_walks)    
    i,j = self._walk(self.grid, *initial, direction, debug=debug)
    self.x.append(i)
    self.y.append(j)
    
    n_steps += 1
    

    while not colide:
      possible_walks, possible_values = self._check_walks(self.grid, i, j)
      
      # Log possible walks
      self.k.append(len(possible_walks))  

      # Check if there is no possible ways to walk 
      # if this is the case then the walker colided and it is terminated
      if np.random.uniform() < self.terminate_probability:
        colide = True
        if len(possible_walks) == 0:
          self.k.pop(-1)    
        break

      if len(possible_walks) == 0:
        colide = True
        
        # Exclude last value because it is 0
        self.k.pop(-1)
        
        break

      direction = self._choose_uniformly(possible_walks)    
      i,j = self._walk(self.grid, i, j, direction, debug=debug)      
      self.x.append(i)
      self.y.append(j)
      
      n_steps += 1            
      
      if plot:
        plt.imshow(self.grid)
        plt.show()
    
    return n_steps