/**
 * This function detects that someone tried to access documentation for fn API function
 * in the old location by accessing `supported_ops.html#nvidia.dali.fn.operator_name`
 * and redirects to the new location of `operations/nvidia.dali.fn.operator_name.html`
 */
function redirect_legacy_op_link() {
  console.log(window.location.pathname);
  if (window.location.pathname.endsWith("supported_ops.html")) {
    var hash = window.location.hash.substring(1);
    if (hash) {
        var current = window.location.href;
        target = current.replace("supported_ops.html", "operations/" + hash + ".html")
        window.location.replace(target);
    }
  }
}
redirect_legacy_op_link();
