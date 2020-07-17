class NavBar extends React.Component{
    constructor(props){
        super(props);
    }
    render(){
        return (
        
            <div className="container-fluid navbar navbar-expand navbar-dark flex-column flex-md-row bd-navbar bg-dark" id={this.props.updater} >
                <a className="navbar-brand" href="#">{this.props.brandname}</a>
                <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                    <span className="navbar-toggler-icon"></span>
                </button>
                <div className="collapse navbar-collapse" id="navbarSupportedContent">
                    <ul className="navbar-nav mr-auto">
                        {this.props.navs.map(nav => <Nav {...nav} key={nav.text}/>)}
                    </ul>
                </div> 
                <Form inline>
                    <FormControl type="text" placeholder="Algo's IP" className="mr-sm-2" id="mod_algo_ip"/>
                    <Button variant="outline-light" className='mr-sm-2' onClick={()=> {add_algo(this.props.update_data_handelr)}}>Add</Button>
                    <Button variant="outline-light" onClick={()=> {remove_algo(this.props.update_data_handelr)}} >Del</Button>
                </Form>
            </div>
        );
    }
}

class Nav extends React.Component{
    constructor(props){
        super(props);
    }
    render(){
        text_color = this.props.text_color ? this.props.text_color : 'text-white';
        
        if (this.props.is_dropdown){
            text_color_class = "dropdown-item " + text_color;
            return (
            <li className="nav-item dropdown nav_menu_align" key={this.props.text}>
            <a className="nav-item nav-link dropdown-toggle mr-md-2 active" href="#" id="bd-versions" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Strategies</a>
            <div className="dropdown-menu" aria-labelledby="navbarDropdown">
            {this.props.dropdown_content.map(content => <a className="dropdown-item" href='#' onClick={content.on_click} key={content.text}>{content.text}</a> )}
            </div>
            </li>
            )

        }

        else {
            var active = this.props.active ? "nav-link active " : "nav-link";
            var text_color_class = active + "" + text_color; 
            var onclick = this.props.on_click;
            return (
            <li className={text_color_class} key={this.props.text}>
            <a className={text_color_class} href='#' onClick={onclick}>{this.props.text}</a> 
            {/* button to change state */}
            </li>
          )
        }
    }
}


