import ConfigParser
import numpy as np

from   math  import pi

###
# Utility object for reading config files
###
class ConfigIter (object):
    '''
    ConfigIter is a utility class that extends the Python ConfigParser in
    several ways:

      1.) Sections are grouped into category types by specifying the section
          names like 'Type: name'.  The ConfigIter object iterates through all
          sections having the type specified in 'Category'.
      2.) Sections can inherit from other sections using a 'Template' key.
      3.) Key values can be Python expressions, evaluated using 'eval'.
    '''

    def __iter__(self):
        return self

    def __init__(self, config, category, defaultname):
        '''
        Create the iterator. `config' should be a ConfigParser object.
        `category' should be the section prefix; sections not beginning with
        'category:' should be ignored.  `category' is case-insensitive.
        '''
        self.Config      = config
        self.Category    = category
        self.DefaultName = defaultname
        self.Sections    = []
        self.Names       = []

        self.Index       = 0

        # Find all matching sections
        for sec in self.Config.sections():
            sp   = sec.split(':')

            if len(sp) < 2 or sp[0].upper() != category.upper():
                continue

            self.Sections.append(sec)
            self.Names.append(sp[1].strip())

    def readSection(self, secName, tgt=None):
        '''
        Read the section specified by `secName', with fancy interpolation and
        recursive templating.  If `tgt' is specified, update that dict.
        otherwise, create a new dict including the interpolated key-value pairs.
        '''

        # If unspecified, create a new output dict.
        if tgt is None:
            tgt = {}

        # Read in template, or default if unspecified.
        if secName != self.DefaultName:
            try:
                tpl = self.Config.get(secName, "Template")
            except ConfigParser.NoOptionError:
                tpl = self.DefaultName

            self.readSection(tpl, tgt)

        # Read in the specified section
        for key, value in self.Config.items(secName):
            try:
                tgt[key] = eval(value)
            except (NameError, SyntaxError):
                tgt[key] = value

        return tgt

    def next(self):
        '''
        Return the next section in the sequence.
        '''
        if self.Index >= len(self.Sections):
            raise StopIteration()

        sec  = self.Sections[self.Index]
        name = self.Names[self.Index]
        conf = self.readSection(sec)
        
        self.Index += 1

        return (conf, name)

