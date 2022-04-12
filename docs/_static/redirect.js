/**
 * Calculate the redirect based on the hash contents or return empty string to indicate no redirect
 */
function hash_to_new_page_name(hash) {
  const base_str = "operations/";
  if (hash.includes("nvidia-dali-")) {
    // someone used the link to the module section, we need to replace the `-` with `.`
    // use regex so we replace all the occurrences
    return base_str + hash.replace(/-/g, ".") + ".html";
  } else if(hash.includes("module-nvidia.dali.")) {
    // side bar navigation used another format for modules
    return base_str + hash.substring("module-".length) + ".html";
  } else if (hash.includes("nvidia.dali.")) {
    // prefix "nvidia.dali." indicates we are dealing with an operator, so we just return it
    return base_str + hash + ".html";
  }
  // Otherwise it's empty
  return "";
}


/**
 * This function detects that someone tried to access documentation for fn API function
 * in the old location by accessing anchor to the dedicated page. Works for following patters:
 * `supported_ops.html#nvidia.dali.fn.operator_name`
 *    -> `operations/nvidia.dali.fn.operator_name.html`
 *
 * `supported_ops.html#nvidia-dali-fn-module_name`
 *    -> `operations/nvidia.dali.fn.module_name.html`
 *
 * `supported_ops.html#module-nvidia.dali.fn.module_name`
 *    -> `operations/nvidia.dali.fn.module_name.html`
 */
function redirect_legacy_op_link() {
  if (window.location.pathname.endsWith("supported_ops.html")) {
    var hash = window.location.hash.substring(1);
    if (hash) {
      var page = hash_to_new_page_name(hash);
      if (page != "") {
        var current = window.location.pathname;
        var target = current.replace("supported_ops.html", page);
        window.location.replace(target);
      }
    }
  }
}
redirect_legacy_op_link();
