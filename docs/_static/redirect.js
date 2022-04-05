// TODO(klecki): doesn't work, how to include it?
// (function () {
//   console.log(window.location.pathname);
//   if (window.location.pathname.endsWith("supported_ops.html")) {
//     console.log("An I here?");
//     /*
//     * Best practice for extracting hashes:
//     * https://stackoverflow.com/a/10076097/151365
//     */
//     var hash = window.location.hash.substring(1);
//     if (hash) {
//         /*
//         * Best practice for javascript redirects:
//         * https://stackoverflow.com/a/506004/151365
//         */
//         var current = window.location.href;
//         target = current.replace("supported_ops.html", "operations" + hash + ".html")
//         window.location.replace(target);
//     }
//   }
// })();