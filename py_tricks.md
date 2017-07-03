####################################
Some frequently used commands
####################################

### List
1) Convert a binary feature in dataframe to one-hot:

   a = df.loc[:,'feature'].values.tolist()
   
   pd.get_dummies(a)

2) Find a certain element x in list a:

   a) a.index(x) -- Only return the first element
   
   b) [i for i, j in enumerate(a) if j == x]
   
3) Delete an element x from list a:

   a) a.remove(x)  -- Only delete the first element
   
   b) del a[x]
   
   c) a.pop(x) -- Could return the value being popped
   
4) Delete a list of elements b from another list a:

   a) [x for x in b if x not in a]
   
   b) c = sorted(set(a) - set(b))

5) About SET:
   
   a) Get unique elements in a list a: set(a)
   
   b) Remove a list b from list a: list(set(a) - set(b))
   
   c) Intersection of two sets a and b: set(a) & set(b)
   
   d) Union of two sets a and b: set(a) | set(b)

### Array



### DataFrame



### Keras package for neual network
