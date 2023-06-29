# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# --- Rust ---
# shared by all users
export RUSTUP_HOME=/datastore/rust/rustup
export PATH=${PATH}:/datastore/rust/cargo/bin

# user-specific
export CARGO_HOME=/datastore/$USER/cargo
export PATH=${PATH}:/datastore/$USER/cargo/bin

# User specific aliases and functions
export PYTHONPATH="$HOME/Repositories"
export PYTHONPATH="$PYTHONPATH:$HOME/Repositories/hfas"
