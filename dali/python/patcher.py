# based on https://github.com/pypa/auditwheel/pull/187 by László Kiss Kollár
import argparse
import lief
import logging

logger = logging.getLogger(__name__)

class Lief():
    def __init__(self, file_name):
        self.file_name = file_name
        self.elf = None
        self.changed = False

    def __enter__(self):
        # see https://github.com/lief-project/LIEF/issues/416
        self.elf = lief.ELF.parse(self.file_name, lief.ELF.DYNSYM_COUNT_METHODS.SECTION)
        self.changed = False
        return self

    def __exit__(self, type, value, traceback):
        if self.changed:
            self.elf.write(self.file_name)
        self.changed = False
        self.elf = None

    def replace_needed(self, so_name, new_so_name):
        name_pairs = zip(so_name, new_so_name)
        names_map = {names[0] : names[1] for names in name_pairs}

        for lib in self.elf.dynamic_entries:
            if lib.tag == lief.ELF.DYNAMIC_TAGS.NEEDED and lib.name in names_map.keys() \
               and lib.name != names_map[lib.name]:
                logger.info("Replacing needed library: %s with %s",
                            lib.name, names_map[lib.name])
                lib.name = names_map[lib.name]
                self.changed = True

        for sym in self.elf.symbols_version_requirement:
            if sym.name in names_map.keys():
                sym.name = names_map[sym.name]
                self.changed = True

    def print_needed(self):
        for lib in self.elf.dynamic_entries:
            if lib.tag == lief.ELF.DYNAMIC_TAGS.NEEDED:
                print(lib.name)

    def set_so_name(self, new_so_name):
        self.changed = True
        # TODO error handling (target might not be a library)
        soname = self.elf.get(lief.ELF.DYNAMIC_TAGS.SONAME)
        soname.name = new_so_name
        logger.info("Setting SONAME to %s", soname)

    def set_rpath(self, libdir):
        self.changed = True
        try:
            rpath = self.elf.get(lief.ELF.DYNAMIC_TAGS.RPATH)
            self.elf.remove(rpath)
        except lief.not_found:
            pass

        try:
            runpath = self.elf.get(lief.ELF.DYNAMIC_TAGS.RUNPATH)
            logger.info("Current RUNPATH: %s", runpath)
            runpath.runpath = libdir
        except lief.not_found:
            logger.info("No RUNPATH found, creating new entry")
            runpath = lief.ELF.DynamicEntryRunPath(libdir)
            self.elf.add(runpath)

        logger.info("Setting new RUNPATH: %s", libdir)

    def print_rpath(self):
        rpath = ""
        try:
            rpath = self.elf.get(lief.ELF.DYNAMIC_TAGS.RUNPATH)
        except lief.not_found:
            pass
        print(rpath)

parser = argparse.ArgumentParser(description='Patchelf replacement based on LIEF')
parser.add_argument('--replace-needed', nargs='*', type=str, metavar=('old_name_1, ..., new_name_1, ...'),
                    help='Replaces a library name that lib links to')
parser.add_argument('--print-needed', type=str, metavar=('target'),
                    help='Print all libraries that target links to')
parser.add_argument('--set-rpath',type=str, metavar=('new_rpath'),
                    help='Sets rpath')
parser.add_argument('--set-soname',type=str, metavar=('new_soname'),
                    help='Sets new soname')
parser.add_argument('--print-rpath', action='store_true',
                    help='Print rpath')
parser.add_argument('--file', type=str, metavar=('file name'),
                    help='Target file')
args = parser.parse_args()

with Lief(args.file) as elf_file:
    if args.replace_needed:
        total_length = len(args.replace_needed)
        if (total_length / 2).is_integer():
            length = total_length // 2
            old_names = args.replace_needed[0:length]
            new_names = args.replace_needed[length:]
            elf_file.replace_needed(old_names, new_names)
        else:
            print("Wrong number of arguments, it should be at least 2*N")
            exit(1)

    if args.print_needed:
        elf_file.print_needed()

    if args.set_rpath:
        elf_file.set_rpath(args.set_rpath)

    if args.set_soname:
        elf_file.set_so_name(args.set_soname)

    if args.print_rpath:
        elf_file.print_rpath()


