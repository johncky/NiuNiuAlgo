var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var NavBar = function (_React$Component) {
    _inherits(NavBar, _React$Component);

    function NavBar(props) {
        _classCallCheck(this, NavBar);

        return _possibleConstructorReturn(this, (NavBar.__proto__ || Object.getPrototypeOf(NavBar)).call(this, props));
    }

    _createClass(NavBar, [{
        key: "render",
        value: function render() {
            var _this2 = this;

            return React.createElement(
                "div",
                { className: "container-fluid navbar navbar-expand navbar-dark flex-column flex-md-row bd-navbar bg-dark", id: this.props.updater },
                React.createElement(
                    "a",
                    { className: "navbar-brand", href: "#" },
                    this.props.brandname
                ),
                React.createElement(
                    "button",
                    { className: "navbar-toggler", type: "button", "data-toggle": "collapse", "data-target": "#navbarSupportedContent", "aria-controls": "navbarSupportedContent", "aria-expanded": "false", "aria-label": "Toggle navigation" },
                    React.createElement("span", { className: "navbar-toggler-icon" })
                ),
                React.createElement(
                    "div",
                    { className: "collapse navbar-collapse", id: "navbarSupportedContent" },
                    React.createElement(
                        "ul",
                        { className: "navbar-nav mr-auto" },
                        this.props.navs.map(function (nav) {
                            return React.createElement(Nav, Object.assign({}, nav, { key: nav.text }));
                        })
                    )
                ),
                React.createElement(
                    Form,
                    { inline: true },
                    React.createElement(FormControl, { type: "text", placeholder: "Algo's IP", className: "mr-sm-2", id: "mod_algo_ip" }),
                    React.createElement(
                        Button,
                        { variant: "outline-light", className: "mr-sm-2", onClick: function onClick() {
                                add_algo(_this2.props.update_data_handelr);
                            } },
                        "Add"
                    ),
                    React.createElement(
                        Button,
                        { variant: "outline-light", onClick: function onClick() {
                                remove_algo(_this2.props.update_data_handelr);
                            } },
                        "Del"
                    )
                )
            );
        }
    }]);

    return NavBar;
}(React.Component);

var Nav = function (_React$Component2) {
    _inherits(Nav, _React$Component2);

    function Nav(props) {
        _classCallCheck(this, Nav);

        return _possibleConstructorReturn(this, (Nav.__proto__ || Object.getPrototypeOf(Nav)).call(this, props));
    }

    _createClass(Nav, [{
        key: "render",
        value: function render() {
            text_color = this.props.text_color ? this.props.text_color : 'text-white';

            if (this.props.is_dropdown) {
                text_color_class = "dropdown-item " + text_color;
                return React.createElement(
                    "li",
                    { className: "nav-item dropdown nav_menu_align", key: this.props.text },
                    React.createElement(
                        "a",
                        { className: "nav-item nav-link dropdown-toggle mr-md-2 active", href: "#", id: "bd-versions", role: "button", "data-toggle": "dropdown", "aria-haspopup": "true", "aria-expanded": "false" },
                        "Strategies"
                    ),
                    React.createElement(
                        "div",
                        { className: "dropdown-menu", "aria-labelledby": "navbarDropdown" },
                        this.props.dropdown_content.map(function (content) {
                            return React.createElement(
                                "a",
                                { className: "dropdown-item", href: "#", onClick: content.on_click, key: content.text },
                                content.text
                            );
                        })
                    )
                );
            } else {
                var active = this.props.active ? "nav-link active " : "nav-link";
                var text_color_class = active + "" + text_color;
                var onclick = this.props.on_click;
                return React.createElement(
                    "li",
                    { className: text_color_class, key: this.props.text },
                    React.createElement(
                        "a",
                        { className: text_color_class, href: "#", onClick: onclick },
                        this.props.text
                    )
                );
            }
        }
    }]);

    return Nav;
}(React.Component);